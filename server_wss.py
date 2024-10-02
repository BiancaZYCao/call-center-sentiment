from datetime import datetime
import json, time
import logging
# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
from fastapi import Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY
from pydantic_settings import BaseSettings
from pydantic import BaseModel, Field
from funasr import AutoModel
import numpy as np
import soundfile as sf
import argparse
import uvicorn
from urllib.parse import parse_qs
import os
import asyncio
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from text_sentiment import text_sentiment_inference

from model_predicate import determine_sentiment, calc_feature_all, selected_feature_name, \
    Boosting_Model_Predication, calculate_final_score, retrieve_probability, CNN_Model_Predication, \
    CNN_Model_Predication_New,  calculate_combine_score, determine_sentiment_category, audio_model_inference

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from TopicModel import TopicModel
from fastapi.middleware.cors import CORSMiddleware
import pytz

# Get current time in Singapore
singapore_tz = pytz.timezone('Asia/Singapore')

# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# from sklearn import set_config
# set_config(assume_finite=True)

# Mute OpenAI logging
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
# Mute httpcore debug messages
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpcore.connection').setLevel(logging.WARNING)
logging.getLogger('httpcore.http11').setLevel(logging.WARNING)


# region ASR: STT & speaker verification
class Config(BaseSettings):
    sv_thr: float = Field(0.4, description="Speaker verification threshold")
    chunk_size_ms: int = Field(100, description="Chunk size in milliseconds")
    sample_rate: int = Field(16000, description="Sample rate in Hz")
    bit_depth: int = Field(16, description="Bit depth")
    channels: int = Field(1, description="Number of audio channels")


config = Config()

emo_dict = {
    "<|HAPPY|>": "😊",
    "<|SAD|>": "😔",
    "<|ANGRY|>": "😡",
    "<|NEUTRAL|>": "",
    "<|FEARFUL|>": "😰",
    "<|DISGUSTED|>": "🤢",
    "<|SURPRISED|>": "😮",
}

event_dict = {
    "<|BGM|>": "🎼",
    "<|Speech|>": "",
    "<|Applause|>": "👏",
    "<|Laughter|>": "😀",
    "<|Cry|>": "😭",
    "<|Sneeze|>": "🤧",
    "<|Breath|>": "",
    "<|Cough|>": "🤧",
}

emoji_dict = {
    "<|nospeech|><|Event_UNK|>": "❓",
    "<|zh|>": "",
    "<|en|>": "",
    "<|yue|>": "",
    "<|ja|>": "",
    "<|ko|>": "",
    "<|nospeech|>": "",
    "<|HAPPY|>": "😊",
    "<|SAD|>": "😔",
    "<|ANGRY|>": "😡",
    "<|NEUTRAL|>": "",
    "<|BGM|>": "🎼",
    "<|Speech|>": "",
    "<|Applause|>": "👏",
    "<|Laughter|>": "😀",
    "<|FEARFUL|>": "😰",
    "<|DISGUSTED|>": "🤢",
    "<|SURPRISED|>": "😮",
    "<|Cry|>": "😭",
    "<|EMO_UNKNOWN|>": "",
    "<|Sneeze|>": "🤧",
    "<|Breath|>": "",
    "<|Cough|>": "😷",
    "<|Sing|>": "",
    "<|Speech_Noise|>": "",
    "<|withitn|>": "",
    "<|woitn|>": "",
    "<|GBG|>": "",
    "<|Event_UNK|>": "",
}

lang_dict = {
    "<|zh|>": "<|lang|>",
    "<|en|>": "<|lang|>",
    "<|yue|>": "<|lang|>",
    "<|ja|>": "<|lang|>",
    "<|ko|>": "<|lang|>",
    "<|nospeech|>": "<|lang|>",
}

emo_set = {"😊", "😔", "😡", "😰", "🤢", "😮"}
event_set = {"🎼", "👏", "😀", "😭", "🤧", "😷", }


def format_str(s):
    for sptk in emoji_dict:
        s = s.replace(sptk, emoji_dict[sptk])
    return s


def format_str_v2(s):
    sptk_dict = {}
    for sptk in emoji_dict:
        sptk_dict[sptk] = s.count(sptk)
        s = s.replace(sptk, "")
    emo = "<|NEUTRAL|>"
    for e in emo_dict:
        if sptk_dict[e] > sptk_dict[emo]:
            emo = e
    for e in event_dict:
        if sptk_dict[e] > 0:
            s = event_dict[e] + s
    s = s + emo_dict[emo]

    for emoji in emo_set.union(event_set):
        s = s.replace(" " + emoji, emoji)
        s = s.replace(emoji + " ", emoji)
    return s.strip()


def format_str_v3(s):
    def get_emo(s):
        return s[-1] if s[-1] in emo_set else None

    def get_event(s):
        return s[0] if s[0] in event_set else None

    s = s.replace("<|nospeech|><|Event_UNK|>", "❓")
    for lang in lang_dict:
        s = s.replace(lang, "<|lang|>")
    s_list = [format_str_v2(s_i).strip(" ") for s_i in s.split("<|lang|>")]
    new_s = " " + s_list[0]
    cur_ent_event = get_event(new_s)
    for i in range(1, len(s_list)):
        if len(s_list[i]) == 0:
            continue
        if get_event(s_list[i]) == cur_ent_event and get_event(s_list[i]) != None:
            s_list[i] = s_list[i][1:]
        # else:
        cur_ent_event = get_event(s_list[i])
        if get_emo(s_list[i]) != None and get_emo(s_list[i]) == get_emo(new_s):
            new_s = new_s[:-1]
        new_s += s_list[i].strip().lstrip()
    new_s = new_s.replace("The.", " ")
    return new_s.strip()


sv_pipeline = pipeline(
    task='speaker-verification',
    model='iic/speech_campplus_sv_zh_en_16k-common_advanced',
    model_revision='v1.0.0'
)

asr_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='iic/SenseVoiceSmall',
    model_revision="master",
    device="cuda:0",
)

model = AutoModel(
    model="fsmn-vad",
    model_revision="v2.0.4",
    disable_pbar=True,  # 禁用进度条显示，通常用于防止在非交互式环境中出现多余的输出。
    max_end_silence_time=200,  # 设置最大结束静音时间（单位：毫秒）。如果在检测过程中静音持续超过这个时间，模型可能会认为语音段结束。
    speech_noise_thres=0.8,  # 语音与噪声之间的阈值，用于区分语音和噪声。值越大，模型越倾向于认为音频是噪声。
    disable_update=True  # 禁用模型的自动更新功能，防止在处理过程中更新模型参数。
)

model_name_emo2vec = "iic/emotion2vec_plus_base"
model_emo2vec = AutoModel(model=model_name_emo2vec)

reg_spks_files = [
    "speaker/agent_0013.wav",
    "speaker/agent_0001.wav",
    "speaker/agent_0007.wav",
    "speaker/agent_0022.wav",
    "speaker/agent_0027.wav",
    "speaker/agent_0028.wav",
]



def reg_spk_init(files):
    reg_spk = {}
    for f in files:
        data, sr = sf.read(f, dtype="float32")
        k, _ = os.path.splitext(os.path.basename(f))
        reg_spk[k] = {
            "data": data,
            "sr": sr,
        }
    return reg_spk


reg_spks = reg_spk_init(reg_spks_files)


def process_vad_audio(audio, sv=True, lang="en"):
    speaker_label = "Client"
    # logger.debug(f"[process_vad_audio] process audio(length: {len(audio)})")
    if not sv:
        return speaker_label, asr_pipeline(audio, language=lang.strip())

    hit = False
    for k, v in reg_spks.items():
        res_sv = sv_pipeline([audio, v["data"]], thr=config.sv_thr)
        # logger.debug(f"[speaker check] {k}: {res_sv}")
        if res_sv["score"] >= config.sv_thr:
            logger.warning(f"[speaker check identified] {k}: score at {res_sv['score']}")
            speaker_label = "Agent"
            break

    return speaker_label, asr_pipeline(audio, language=lang.strip())

# endregion

async def async_save_and_infer_emotion(wav_file_path, vad_audio_chunk, sample_rate):
    await asyncio.to_thread(
        sf.write, wav_file_path, vad_audio_chunk, sample_rate, format='WAV', subtype='PCM_16'
    )
    # result = await asyncio.to_thread(
    #     model.generate,
    #     wav_file_path,
    #     output_dir="./outputs",
    #     granularity="utterance",
    #     extract_embedding=False
    # )
    # logger.debug("[EMO2VEC] emotion2vec result: ", result)
    # return result


app = FastAPI()

# 设置允许跨域访问的源
origins = [
    "http://localhost:63342",  # 允许的前端地址
    "http://127.0.0.1:63342",  # 也可以添加其他需要的地址
]
# 设置跨域中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def custom_exception_handler(request: Request, exc: Exception):
    logger.error("Exception occurred", exc_info=True)
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


# Define the response model
class TranscriptionResponse(BaseModel):
    code: int
    msg: str
    data: str
    type: str  # e.g., 'stt' (speech-to-text), 'sentiment', 'score'
    timestamp: str  # Include timestamp as an ISO format string
    speaker_label: str = ""  # Speaker label

class AnalysisResponse(BaseModel):
    code: int = 0
    msg: str = 'success'
    data: str
    type: str  # e.g. 'text-sentiment', 'topic', 'audio-sentiment'
    timestamp: str  # Include timestamp as an ISO format string

# 全局变量
final_score_list = []  # 存储所有的最终得分
cache = {}  # 接收客户端传输的二进制音频数据
# Create a global queue for passing STT results from WebSocket 1 to WebSocket 2
stt_queue = asyncio.Queue()
tm = TopicModel()
end_time_list = []  # 存储所有的结束时间
# 实时音频流的语音识别和说话人验证
@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket_trans: WebSocket):
    try:
        # 1. websocket 连接处理
        query_params = parse_qs(websocket_trans.scope['query_string'].decode())
        sv = query_params.get('sv', ['false'])[0].lower() in ['true', '1', 't', 'y', 'yes']
        lang = query_params.get('lang', ['en'])[0].lower()

        await websocket_trans.accept()  # 接受 WebSocket 连接，开始与客户端通信

        # 2. 音频块大小的计算
        # 计算每个音频块的大小（以字节为单位），用于切分音频数据流。
        chunk_size = int(config.chunk_size_ms * config.sample_rate * config.channels * (config.bit_depth // 8) / 1000)

        # 3.音频缓冲处理
        audio_buffer = np.array([])  # 存储接收到的原始音频数据
        audio_vad = np.array([])  # 用于存储语音活动检测（VAD）后的音频片段

        cache = {}  # 接收客户端传输的二进制音频数据
        cache_text_client = ""

        # 初始化语音活动的开始和结束时间的标记
        last_vad_beg = last_vad_end = -1

        # 初始化偏移量，用于跟踪语音活动检测的位置。
        offset = 0

        # 4.  接收音频数据并进行处理
        while True:
            data = await websocket_trans.receive_bytes()  # 接收客户端传输的二进制音频数据
            # logger.debug(f"received {len(data)} bytes")

            audio_buffer = np.append(audio_buffer, np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0)

            while len(audio_buffer) >= chunk_size:
                # 从audio_buffer 的开头到chunk_size, 提取大小为chunk size 的音频快
                chunk = audio_buffer[:chunk_size]  # chunk是一个包含浮点数的 NumPy 数组，每个值代表一个音频样本的振幅 ,[-1.0, 1.0]
                # 删除audio_buffer中之前被提取过的部分代码块
                audio_buffer = audio_buffer[chunk_size:]  # audio_buffer 只保留尚未处理的音频数据
                # 将刚提取到的chunk 添加到audio_vad数组中
                audio_vad = np.append(audio_vad, chunk)

                # 5. VAD 推断音频块
                res = model.generate(input=chunk, cache=cache, is_final=False, chunk_size=config.chunk_size_ms)
                # 6. 检查推理结果
                if len(res[0]["value"]):  # 如果result中有值
                    vad_segments = res[0]["value"]
                    # 7. 提取语音活动时间段
                    for segment in vad_segments:
                        if segment[0] > -1:  # speech begin
                            last_vad_beg = segment[0]
                        if segment[1] > -1:  # speech end
                            last_vad_end = segment[1]
                        if last_vad_beg > -1 and last_vad_end > -1:
                            # logger.debug(f"vad segment: {[last_vad_beg, last_vad_end]}")
                            # try to use real timestamps
                            original_vad_timeline = {
                                "start_time_relative": segment[0] / 1000,
                                "end_time_relative": segment[1] / 1000
                            }
                            logger.debug(f"vad segment ms coordinates: {[last_vad_beg/1000, last_vad_end/1000]}")
                            start = time.time()
                            last_vad_beg -= offset
                            last_vad_end -= offset
                            offset += last_vad_end
                            beg = int(last_vad_beg * config.sample_rate / 1000)  # 语音活动开始位置
                            end = int(last_vad_end * config.sample_rate / 1000)  # 语音活动的结束位置

                            # 获取经过 VAD 处理的音频块 - 20240904
                            vad_audio_chunk = audio_vad[beg:end]

                            # 调用process_vad_audio()函数对这些片段进一步处理 --- old
                            speaker_label, transcript_result = process_vad_audio(audio_vad[beg:end], sv, lang)  # todo: async
                            print("[TIME] - STT takes {:.2f} seconds".format(time.time() - start))
                            # logger.debug(f"[process_vad_audio] {speaker_label}: {transcript_result}")

                            # Parameters for sliding window
                            window_size_seconds = 5  # 5 seconds window size
                            stride_seconds = 2.5  # 2.5 seconds stride
                            # Convert window size and stride to samples
                            window_size_samples = int(window_size_seconds * config.sample_rate)
                            stride_samples = int(stride_seconds * config.sample_rate)

                            # Variable to store the last valid score and sentiment
                            last_valid_audio_score = None
                            last_valid_audio_class = None

                            # logger.debug(f"VAD Chunk duration: {len(vad_audio_chunk)/16000}")
                            # Calculate how many inference steps are required
                            inference_time_required = len(vad_audio_chunk) // (window_size_samples // 2) + 1

                            # Iterate over the long vad_audio_chunk with sliding windows
                            for i in range(inference_time_required):
                                # Calculate the start and end indices for each chunk
                                start = i * (window_size_samples // 2)
                                end_window = start + window_size_samples

                                # Extract the chunk
                                chunk = vad_audio_chunk[start:end]
                                # logger.debug(f"start to process chunk:{start}-{end_window}")

                                # Run audio inference on the chunk
                                final_audio_score, final_audio_class = audio_model_inference(chunk)

                                # Error handling: If the inference result is None, use the last valid score and class
                                if final_audio_score is None or final_audio_class is None:
                                    final_audio_score = last_valid_audio_score if last_valid_audio_score is not None else 0
                                    final_audio_class = last_valid_audio_class if last_valid_audio_class is not None else "Neutral sentiment"
                                    logger.warning(
                                        f"Inference failed for chunk, using last valid score: {final_audio_score}, class: {final_audio_class}")
                                else:
                                    last_valid_audio_score = final_audio_score  # Update last valid score
                                    last_valid_audio_class = final_audio_class  # Update last valid class

                                # Calculate relative start and end times for this chunk
                                chunk_start_time_relative = last_vad_beg / 1000 + (start / config.sample_rate)
                                chunk_end_time_relative = last_vad_beg / 1000 + (end_window / config.sample_rate)
                                end_time_offset = (offset / 1000 - len(vad_audio_chunk) / config.sample_rate +
                                                   chunk_start_time_relative)

                                # Append results to timeline and score list
                                if start == 0:
                                    final_score_list.append(final_audio_score)
                                    end_time_list.append(offset / 1000 - len(vad_audio_chunk) / config.sample_rate)
                                final_score_list.append(final_audio_score)
                                end_time_list.append(end_time_offset)
                                logger.warning(f"[DEBUG] AUDIO Result: {chunk_start_time_relative}, "
                                               f"{chunk_end_time_relative}, {end_time_offset} : "
                                               f"{final_audio_score} ")

                                # Create response for this audio chunk
                                response_audio_data = {
                                    "final_score": final_audio_score,
                                    "final_sentiment_3": final_audio_class
                                }
                                response_audio_data_str = json.dumps(response_audio_data)

                                # Optionally send back this response via WebSocket or handle further
                                logger.warning(f"Audio inference result: {response_audio_data_str}")
                                response_audio = TranscriptionResponse(
                                    code=0,
                                    msg=f"success",
                                    data=response_audio_data_str,
                                    type="audio_sentiment",
                                    timestamp=datetime.now(singapore_tz).isoformat(),
                                    speaker_label=speaker_label
                                )
                                await websocket_trans.send_json(response_audio.model_dump())

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

                                # if speaker_label == "Agent":
                                #     end_time_list.append(offset / 1000)
                                #     final_score_list.append(None)

                                if speaker_label == "Client":
                                    # text sentiment - send to queue
                                    result_text_dict = {
                                        "stt_text": result_text,
                                        "timeline_data":  {
                                            "start_time_relative": last_vad_beg / 1000,
                                            "end_time_relative": last_vad_end / 1000
                                        }
                                    }
                                    await stt_queue.put(result_text_dict)

                                    # # Call the asynchronous function to save the file
                                    # wav_file_path = "./temp_chunk.wav"
                                    # await async_save_and_infer_emotion(wav_file_path, vad_audio_chunk, 16000)



                            audio_vad = audio_vad[end:]  # 已经处理过的片段移除，保留未处理的部分
                            last_vad_beg = last_vad_end = -1  # 重置 VAD 片段标记

    except WebSocketDisconnect:
        logger.warning("WebSocket Transcribe disconnected")
        logger.warning(f"[END] final_score_list: {final_score_list}")
        logger.warning(f"[END] end_time_list: {end_time_list}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        await websocket_trans.close()
    finally:
        audio_buffer = np.array([])
        audio_vad = np.array([])
        cache.clear()
        logger.info("Cleaned up resources after WebSocket disconnect")


@app.websocket("/ws/analysis")
async def websocket_analysis_endpoint(websocket_analysis: WebSocket):
    await websocket_analysis.accept()
    try:
        cache_text_client = ""
        while True:
            # Wait to get STT result from the queue
            stt_result_dict = await stt_queue.get()  # Waits until an STT result is available
            stt_result_text = stt_result_dict["stt_text"]

            print(f"Processing sentiment for: {stt_result_text}")
            received_at = datetime.now(singapore_tz).isoformat()
            cache_text_client += " " + stt_result_text
            if len(cache_text_client.split(' ')) >= 7:
                # Sentiment on Text
                text_sentiment_result = text_sentiment_inference(cache_text_client)
                response_sentiment = AnalysisResponse(
                    data=text_sentiment_result,
                    type="text_sentiment",
                    timestamp=received_at
                )
                await websocket_analysis.send_json(response_sentiment.model_dump())

                # Perform topic modeling as well
                topic_results = tm.getTopics(cache_text_client)
                topic_results_str = json.dumps(list(set(topic_results)))
                response_topic = AnalysisResponse(
                    data=topic_results_str,
                    type="topics",
                    timestamp=received_at
                )
                # Send topic modeling results back to the client
                await websocket_analysis.send_json(response_topic.model_dump())

                # Perform topic modeling and get questions for each topic
                topics_and_questions = tm.getTopicsAndQuestions()
                topics_and_questions_str = json.dumps(topics_and_questions)
                response_topic_and_questions = AnalysisResponse(
                    data=topics_and_questions_str,
                    type="topicsAndQuestions",  # Change type to "topicsAndQuestions"
                    timestamp=received_at
                )
                # Send topics and questions back to the client
                await websocket_analysis.send_json(response_topic_and_questions.model_dump())
                cache_text_client = ""  # reset
    except WebSocketDisconnect:
        logger.warning("WebSocket Analysis disconnected")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        await websocket_analysis.close()
    finally:
        # reset list
        final_score_list.clear()
        end_time_list.clear()
        logger.info("Cleaned up resources after WebSocket disconnect")

# 更新折线图
@app.post("/update-chart/")
async def update_chart():
    try:
        if not end_time_list or not final_score_list:  # Check if the lists are empty
            return {"end_time": None, "final_score": None}

        response = {
            "end_time": end_time_list,
            "final_score": final_score_list
        }
        return response

    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating chart: {str(e)}")




# run server
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FastAPI app with a specified port.")
    parser.add_argument('--port', type=int, default=8000, help='Port number to run the FastAPI app on.')
    # parser.add_argument('--certfile', type=str, default='path_to_your_certfile', help='SSL certificate file')
    # parser.add_argument('--keyfile', type=str, default='path_to_your_keyfile', help='SSL key file')
    args = parser.parse_args()

    # uvicorn.run(app, host="0.0.0.0", port=args.port, ssl_certfile=args.certfile, ssl_keyfile=args.keyfile)
    uvicorn.run(app, host="127.0.0.1", port=args.port)
