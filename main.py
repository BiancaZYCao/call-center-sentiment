""" Main Server Script - initialize with websocket server """
from datetime import datetime
import json
import time
import logging
import traceback
import argparse
import pytz
import uvicorn
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi import Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY
# for ASP import
from funasr import AutoModel
import numpy as np
from urllib.parse import parse_qs
# self develop module import
from schema.response import TranscriptionResponse, AnalysisResponse
from schema.request import QuestionRequest
from utils.text_formatting import format_str_v3
from utils.score_adjust import adjust_audio_scores, update_final_scores
from utils.speaker_recognition import asr_config, vad_model, recognize_agent_speaker_after_vad
from model_inference.text_sentiment import text_sentiment_inference
from model_inference.text_analysis import TopicModel
from model_inference.speech_sentiment import audio_model_inference

# Get current time in Singapore
singapore_tz = pytz.timezone('Asia/Singapore')
# Mute OpenAI logging
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
# Mute httpcore debug messages
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpcore.connection').setLevel(logging.WARNING)
logging.getLogger('httpcore.http11').setLevel(logging.WARNING)
# option to use larger emotion model - cause lagging
# model_name_emo2vec = "iic/emotion2vec_plus_base"
# model_emo2vec = AutoModel(model=model_name_emo2vec)


# region start FastAPI Engine
app = FastAPI()

# CORS
origins = [
    "http://localhost:63342",
    "http://127.0.0.1:63342",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=3600,  # Cache the preflight response for 1 hour
)


@app.exception_handler(Exception)
async def custom_exception_handler(request: Request, exc: Exception):
    logging.error("Exception occurred", exc_info=True)
    if isinstance(exc, HTTPException):
        status_code = exc.status_code
        message = exc.detail
        data = ""
    elif isinstance(exc, RequestValidationError):
        status_code = HTTP_422_UNPROCESSABLE_ENTITY
        message = "Validation error: " + str(exc.errors())
        data = ""
    else:
        status_code = 500
        message = "Internal server error: " + str(exc)
        data = ""

    return JSONResponse(
        status_code=status_code,
        content=TranscriptionResponse(
            code=status_code,
            msg=message,
            data=data,
            type='error',
            timestamp=datetime.now(singapore_tz).isoformat(),  # UTC timestamp
        ).model_dump()
    )
# endregion


# region Global variables
final_score_list = []
cache = {}
# Create a global queue for passing STT results from WebSocket 1 to WebSocket 2
stt_queue = asyncio.Queue()
tm = TopicModel()
end_time_list = []
cache_text_client = ""
audio_score_list = []
timeline_data_list = []
lock_score_list = asyncio.Lock()
lock_tm = asyncio.Lock()
# endregion


# Real-time ASR and Speaker detection
@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket_trans: WebSocket):
    """
    websocket endpoint - transcribe
    listening to streaming audio signal
    determine speaker ID, content, and audio sentiment inference
    pass content to UI (websocket) and text processing module (queue)
    """
    global final_score_list, end_time_list, cache
    try:
        # 1. websocket connection
        query_params = parse_qs(websocket_trans.scope['query_string'].decode())
        sv = query_params.get('sv', ['false'])[0].lower() in ['true', '1', 't', 'y', 'yes']
        lang = query_params.get('lang', ['en'])[0].lower()

        await websocket_trans.accept()

        # 2. calculate audio chunk
        # Calculate the size of each audio chunk (in bytes) for splitting the audio data stream.
        chunk_size = int(asr_config.chunk_size_ms * asr_config.sample_rate * asr_config.channels * (asr_config.bit_depth // 8) / 1000)

        # 3. audio cache processing
        audio_buffer = np.array([])
        audio_vad = np.array([])
        cache = {}
        # initial tag
        last_vad_beg = last_vad_end = -1
        # initial offset
        offset = 0

        # 4.  handle audio to VAD ASR
        while True:
            data = await websocket_trans.receive_bytes()

            audio_buffer = np.append(audio_buffer, np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0)

            while len(audio_buffer) >= chunk_size:
                chunk = audio_buffer[:chunk_size]  # chunk a  NumPy array，[-1.0, 1.0]
                audio_buffer = audio_buffer[chunk_size:]
                audio_vad = np.append(audio_vad, chunk)

                # 5. VAD
                res = vad_model.generate(input=chunk, cache=cache, is_final=False, chunk_size=asr_config.chunk_size_ms)
                # 6. check inference outcome
                if len(res[0]["value"]):
                    vad_segments = res[0]["value"]
                    # 7. audio activities timeline
                    for segment in vad_segments:
                        if segment[0] > -1:  # speech begin
                            last_vad_beg = segment[0]
                        if segment[1] > -1:  # speech end
                            last_vad_end = segment[1]
                        if last_vad_beg > -1 and last_vad_end > -1:
                            logging.debug(f"[VAD] segment: {[last_vad_beg, last_vad_end]}")
                            # use real on-going timestamps
                            logging.debug(f"[VAD] segment ms coordinates: {[last_vad_beg/1000, last_vad_end/1000]}")
                            start = time.time()
                            last_vad_beg -= offset
                            last_vad_end -= offset
                            offset += last_vad_end
                            beg = int(last_vad_beg * asr_config.sample_rate / 1000)
                            end = int(last_vad_end * asr_config.sample_rate / 1000)

                            vad_audio_chunk = audio_vad[beg:end]

                            speaker_label, transcript_result = recognize_agent_speaker_after_vad(
                                audio_vad[beg:end], sv, lang)  # todo: async
                            logging.info("[TIME] - STT takes {:.2f} seconds".format(time.time() - start))
                            logging.debug(f"[STT] result {speaker_label}: {transcript_result}")

                            # Parameters for sliding window
                            window_size_seconds = 5  # 5 seconds window size
                            stride_seconds = 2.5  # 2.5 seconds stride
                            # Convert window size and stride to samples
                            window_size_samples = int(window_size_seconds * asr_config.sample_rate)
                            stride_samples = int(stride_seconds * asr_config.sample_rate)

                            # Append results to timeline and score list
                            temp_score_list, temp_end_time_list = [], []
                            # Variable to store the last valid score and sentiment
                            last_valid_audio_score = None
                            last_valid_audio_class = None

                            logging.debug(f"VAD Chunk duration: {len(vad_audio_chunk)/16000}")
                            # Calculate how many inference steps are required
                            inference_time_required = len(vad_audio_chunk) // (window_size_samples // 2) + 1
                            final_audio_score, final_audio_class = 0, "Neutral sentiment"

                            # Iterate over the long vad_audio_chunk with sliding windows
                            for i in range(inference_time_required):
                                # Calculate the start and end indices for each chunk
                                start = i * stride_samples
                                end_window = start + window_size_samples

                                # Extract the chunk
                                chunk = vad_audio_chunk[start:end]
                                logging.debug(f"start to process chunk:{start}-{end_window}")

                                # Run audio inference on the chunk
                                final_audio_score, final_audio_class = None, None
                                if len(chunk) > int(0.25 * asr_config.sample_rate):
                                    final_audio_score, final_audio_class = audio_model_inference(chunk)

                                # Error handling: If the inference result is None, use the last valid score and class
                                if final_audio_score is None or final_audio_class is None:
                                    final_audio_score = last_valid_audio_score if last_valid_audio_score is not None else 0
                                    final_audio_class = last_valid_audio_class if last_valid_audio_class is not None else "Neutral sentiment"
                                    logging.warning(
                                        f"Inference failed for chunk, using last valid score: {final_audio_score}, class: {final_audio_class}")
                                else:
                                    last_valid_audio_score = final_audio_score  # Update last valid score
                                    last_valid_audio_class = final_audio_class  # Update last valid class

                                # Calculate relative start and end times for this chunk
                                chunk_start_time_relative = last_vad_beg / 1000 + (start / asr_config.sample_rate)
                                chunk_end_time_relative = last_vad_beg / 1000 + (end_window / asr_config.sample_rate)
                                end_time_offset = (offset / 1000 - len(vad_audio_chunk) / asr_config.sample_rate +
                                                   chunk_start_time_relative)


                                if start == 0 and len(chunk) > int(2.5 * asr_config.sample_rate):
                                    temp_score_list.append(final_audio_score)
                                    temp_end_time_list.append(offset / 1000 - len(vad_audio_chunk) / asr_config.sample_rate)
                                temp_score_list.append(final_audio_score)
                                temp_end_time_list.append(end_time_offset)
                                logging.debug(f"[DEBUG] AUDIO Result: {chunk_start_time_relative}, "
                                               f"{chunk_end_time_relative}, {end_time_offset} : "
                                               f"{final_audio_score} ")


                            if transcript_result is not None:
                                result_text = format_str_v3(transcript_result[0]['text'])
                                # speech to text transcript results
                                response = TranscriptionResponse(
                                    code=0,
                                    msg=f"success",
                                    data=result_text,
                                    type="STT",
                                    timestamp=datetime.now(singapore_tz).isoformat(),
                                    speaker_label=speaker_label
                                )
                                await websocket_trans.send_json(response.model_dump())
                                # if agent, append null scores to make the curve break
                                if speaker_label == "Agent":
                                    async with lock_score_list:
                                        end_time_list += [round(x, 2) for x in temp_end_time_list if x is not None]
                                        final_score_list += [None] * len(temp_end_time_list)
                                # if client, append scores in the list
                                if speaker_label == "Client":
                                    async with lock_score_list:
                                        final_score_list += [round(x, 3) for x in temp_score_list if x is not None]
                                        end_time_list += [round(x, 2) for x in temp_end_time_list if x is not None]
                                    # Create response for this audio chunk
                                    response_audio_data = {
                                        "final_score": np.average(temp_score_list),
                                        "final_sentiment_3": final_audio_class
                                    }
                                    response_audio_data_str = json.dumps(response_audio_data)

                                    # Optionally send back this response via WebSocket or handle further
                                    logging.debug(f"Audio inference result: {response_audio_data_str}")
                                    response_audio = TranscriptionResponse(
                                        code=0,
                                        msg=f"success",
                                        data=response_audio_data_str,
                                        type="audio_sentiment",
                                        timestamp=datetime.now(singapore_tz).isoformat(),
                                        speaker_label=speaker_label
                                    )
                                    await websocket_trans.send_json(response_audio.model_dump())
                                    # text sentiment - send to queue
                                    result_text_dict = {
                                        "stt_text": result_text,
                                        "audio_score_data": temp_score_list,
                                        "timeline_data": temp_end_time_list
                                    }
                                    await stt_queue.put(result_text_dict)

                                    # # Call the function to inference emotion
                                    # rec_result = model_emo2vec.generate(chunk, granularity="utterance", extract_embedding=False)

                            audio_vad = audio_vad[end:]
                            last_vad_beg = last_vad_end = -1

    except WebSocketDisconnect as e:
        logging.warning(f"WebSocket Transcribe disconnected with close code: {e.code}")
        logging.warning(f"[END] final_score_list: {final_score_list}")
        logging.warning(f"[END] end_time_list: {end_time_list}")
    except Exception as e:
        logging.error(f"Unexpected error at ws/transcribe: {e} {traceback.format_exc()}")
        # await websocket_trans.close()  # keep connection open
    finally:
        audio_buffer.resize(0)
        audio_vad.resize(0)
        cache.clear()
        logging.info("Cleaned up resources after WebSocket disconnect")


@app.websocket("/ws/analysis")
async def websocket_analysis_endpoint(websocket_analysis: WebSocket):
    """
    websocket channel to do text part analysis
    correct audio score list segment if conflicts
    Get speech to text input fromm queue
    """
    await websocket_analysis.accept()
    global cache_text_client, audio_score_list, timeline_data_list, final_score_list, end_time_list
    try:
        cache_text_client = ""
        cache_text_topic_list = []
        audio_score_list = []
        timeline_data_list = []
        while True:
            # Wait to get STT result from the queue
            stt_result_dict = await stt_queue.get()  # Waits until an STT result is available
            stt_result_text = stt_result_dict["stt_text"]
            audio_score_list += stt_result_dict["audio_score_data"]
            timeline_data_list += stt_result_dict["timeline_data"]
            average_audio_score = round(sum(audio_score_list) / len(audio_score_list), 2)

            received_at = datetime.now(singapore_tz).isoformat()
            cache_text_client += stt_result_text + " "

            if len(cache_text_client.split(" ")) >= 10:
                logging.info(f"[TEXT] Processing sentiment for: {cache_text_client}")
                text_sentiment_result = text_sentiment_inference(cache_text_client)
                response_sentiment = AnalysisResponse(
                    data=text_sentiment_result,
                    type="text_sentiment",
                    timestamp=str(received_at)
                )
                await websocket_analysis.send_json(response_sentiment.model_dump())

                # adjust sentiment score
                adjusted_audio_scores = adjust_audio_scores(audio_score_list, text_sentiment_result)
                if adjusted_audio_scores != audio_score_list:
                    async with lock_score_list:
                        logging.debug("[ADJ] adjusting final score list into: %s", adjusted_audio_scores)
                        final_score_list = update_final_scores(final_score_list, end_time_list,
                                                               timeline_data_list, adjusted_audio_scores)

                cache_text_topic_list.append(cache_text_client)
                if len(cache_text_topic_list) >= 5:
                    cache_text_topic_list.pop(0)
                cache_text_topic = " ".join(cache_text_topic_list)
                # perform topic modeling (intention)
                async with lock_tm:
                    topic_results = tm.find_topics(cache_text_topic)
                logging.debug(f"[Topic]: {topic_results}")
                topic_results_str = json.dumps(list(set(topic_results)))
                response_topic = AnalysisResponse(
                    data=topic_results_str,
                    type="topics",
                    timestamp=received_at
                )
                await websocket_analysis.send_json(response_topic.model_dump())

                # perform topic and question generated
                async with lock_tm:
                    topics_and_questions = tm.getTopicsAndQuestions()
                topics_and_questions_str = json.dumps(topics_and_questions)
                response_topic_and_questions = AnalysisResponse(
                    data=topics_and_questions_str,
                    type="topicsAndQuestions",
                    timestamp=received_at
                )
                await websocket_analysis.send_json(response_topic_and_questions.model_dump())

                cache_text_client = ""  # reset
                audio_score_list = []
                timeline_data_list = []
    except WebSocketDisconnect as e:
        logging.warning(f"WebSocket Analysis disconnected with close code: {e.code}")
    except Exception as e:
        logging.error(f"Unexpected error at ws/analysis: {e} {traceback.format_exc()}")
        # await websocket_analysis.close() # keep connection open
        pass
    finally:
        # reset list
        final_score_list.clear()
        end_time_list.clear()
        logging.info("Cleaned up resources after WebSocket disconnect")


# update line charts
@app.get("/update-chart/")
async def update_chart():
    """ GET API for updating chart on UI curve """
    try:
        if not end_time_list or not final_score_list:  # Check if the lists are empty
            return {"end_time": None, "final_score": None}
        response = {
            "end_time": end_time_list,
            "final_score": final_score_list
        }
        return response

    except Exception as e:
        logging.error(f"Error processing audio: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating chart: {str(e)}")


@app.post("/get-answer")
async def get_rag_answer(request: QuestionRequest):
    """ GET APT for RAG response """
    if request.type != "selected_question":
        raise HTTPException(status_code=400, detail="Invalid request type")

    selected_question = request.data
    loading_id = request.loadingId

    # Log the received question
    logging.info(f"Received request for RAG answer: {selected_question}, loadingId: {loading_id}")

    # Fetch the answer from the tm model
    try:
        async with lock_tm:  # Use the lock to avoid concurrency issues with the tm instance
            res = tm.gen_response_for_questions_w_RAG(selected_question)

        # Prepare and return the response
        response = {
            "type": "question_answer",
            "data": res,
            "loadingId": loading_id  # Pass the loadingId back to the frontend
        }
        logging.info(f"Sending RAG result: {res}")
        return response
    except Exception as e:
        # Handle any errors that occur during fetching the answer
        logging.error(f"Error fetching answer: {e}")
        raise HTTPException(status_code=500, detail="Error fetching answer")


# run server
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FastAPI app with a specified port.")
    parser.add_argument('--port', type=int, default=8000, help='Port number to run the FastAPI app on.')
    # parser.add_argument('--certfile', type=str, default='path_to_your_certfile', help='SSL certificate file')
    # parser.add_argument('--keyfile', type=str, default='path_to_your_keyfile', help='SSL key file')
    args = parser.parse_args()

    # uvicorn.run(app, host="0.0.0.0", port=args.port, ssl_certfile=args.certfile, ssl_keyfile=args.keyfile)
    uvicorn.run(app, host="127.0.0.1", port=args.port, timeout_keep_alive=1200)
