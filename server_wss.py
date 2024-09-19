from datetime import datetime
from datetime import timedelta
import logging
# Set up logging
logging.basicConfig(level=logging.DEBUG)
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
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from text_sentiment import text_sentiment_inference

from model_predicate import determine_sentiment, calc_feature_all, selected_feature_name, \
    Boosting_Model_Predication, calculate_final_score, retrieve_probability, CNN_Model_Predication, \
    CNN_Model_Predication_New,  calculate_combine_score, determine_sentiment_category

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from TopicModel import TopicModel


from fastapi.middleware.cors import CORSMiddleware

# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#
# from sklearn import set_config
# set_config(assume_finite=True)


class Config(BaseSettings):
    sv_thr: float = Field(0.29, description="Speaker verification threshold")
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
    max_end_silence_time=900,  # 设置最大结束静音时间（单位：毫秒）。如果在检测过程中静音持续超过这个时间，模型可能会认为语音段结束。
    speech_noise_thres=0.8,  # 语音与噪声之间的阈值，用于区分语音和噪声。值越大，模型越倾向于认为音频是噪声。
    disable_update=True  # 禁用模型的自动更新功能，防止在处理过程中更新模型参数。
)

reg_spks_files = [
    # "speaker/speaker1_a_cn_16k.wav"
    "speaker/agent_0013.wav",
    # "speaker/client_4366.wav"
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





def process_vad_audio(audio, sv=True, lang="auto"):
    # update at 20240917
    # speaker_label = "client"
    logger.debug(f"[process_vad_audio] process audio(length: {len(audio)})")
    if not sv:
        return asr_pipeline(audio, language=lang.strip())

    hit = False
    for k, v in reg_spks.items():
        res_sv = sv_pipeline([audio, v["data"]], thr=config.sv_thr)
        logger.debug(f"[speaker check] {k}: {res_sv}")
        if res_sv["score"] >= config.sv_thr:
            hit = True
            # update at 20240917
            # logger.debug(f"[speaker check identified] {k}: score at {res_sv['score']}")
            # speaker_label = k.split("_")[0]
            # break

    return asr_pipeline(audio, language=lang.strip()) if hit else None

    # update at 20240917
    # return speaker_label, asr_pipeline(audio, language=lang.strip())


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
            data=data
        ).model_dump()
    )


# Define the response model
class TranscriptionResponse(BaseModel):
    code: int
    msg: str
    data: str


# 全局变量
timeline_data = []  # Store timestamp
# 实时音频流的语音识别和说话人验证
@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    try:

        # 1. websocket 连接处理
        query_params = parse_qs(websocket.scope['query_string'].decode())
        sv = query_params.get('sv', ['false'])[0].lower() in ['true', '1', 't', 'y', 'yes']
        lang = query_params.get('lang', ['en'])[0].lower()

        await websocket.accept()  # 接受 WebSocket 连接，开始与客户端通信

        # 2. 音频块大小的计算
        # 计算每个音频块的大小（以字节为单位），用于切分音频数据流。
        chunk_size = int(config.chunk_size_ms * config.sample_rate * config.channels * (config.bit_depth // 8) / 1000)

        # 3.音频缓冲处理
        audio_buffer = np.array([])  # 存储接收到的原始音频数据
        audio_vad = np.array([])  # 用于存储语音活动检测（VAD）后的音频片段

        cache = {}  # 接收客户端传输的二进制音频数据

        # 初始化语音活动的开始和结束时间的标记
        last_vad_beg = last_vad_end = -1

        # 初始化偏移量，用于跟踪语音活动检测的位置。
        offset = 0

        # 4.  接收音频数据并进行处理
        while True:
            data = await websocket.receive_bytes()  # 接收客户端传输的二进制音频数据
            logger.debug(f"received {len(data)} bytes")

            audio_buffer = np.append(audio_buffer, np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0)

            while len(audio_buffer) >= chunk_size:
                # 从audio_buffer 的开头到chunk_size, 提取大小为chunk size 的音频快
                chunk = audio_buffer[:chunk_size]  # chunk是一个包含浮点数的 NumPy 数组，每个值代表一个音频样本的振幅 ,[-1.0, 1.0]
                # 删除audio_buffer中之前被提取过的部分代码块
                audio_buffer = audio_buffer[chunk_size:]  # audio_buffer 只保留尚未处理的音频数据
                # 将刚提取到的chunk 添加到audio_vad数组中
                audio_vad = np.append(audio_vad, chunk)

                # with open("debug.pcm", "ab") as f:
                #     f.write(np.int16(chunk * 32767).tobytes())   # 转化为16位证书， NumPy 数组转换为字节序列 #[-32767, 32767]
                # 5. VAD 推断音频块
                res = model.generate(input=chunk, cache=cache, is_final=False, chunk_size=config.chunk_size_ms)
                # 6. 检查推理结果
                logger.debug(f"vad inference: {res}")
                if len(res[0]["value"]):  # 如果result中有值
                    vad_segments = res[0]["value"]
                    # 7. 提取语音活动时间段
                    for segment in vad_segments:
                        if segment[0] > -1:  # speech begin
                            last_vad_beg = segment[0]
                        if segment[1] > -1:  # speech end
                            last_vad_end = segment[1]
                        if last_vad_beg > -1 and last_vad_end > -1:
                            logger.debug(f"vad segment: {[last_vad_beg, last_vad_end]}")
                            last_vad_beg -= offset
                            last_vad_end -= offset
                            offset += last_vad_end
                            beg = int(last_vad_beg * config.sample_rate / 1000)  # 语音活动开始位置
                            end = int(last_vad_end * config.sample_rate / 1000)  # 语音活动的结束位置

                            # 获取经过 VAD 处理的音频块 - 20240904
                            wav_file_path = "./temp_chunk.wav"
                            vad_audio_chunk = audio_vad[beg:end]
                            sf.write(wav_file_path, vad_audio_chunk, 16000, format='WAV',
                                     subtype='PCM_16')  # 使用 soundfile 保存音频

                            # 保存每个VAD的开始时间，结束时间 - 20240912
                            timeline_data.append({
                                "start_time_relative": last_vad_beg/1000,
                                "end_time_relative": last_vad_end/1000
                            })

                            # 调用process_vad_audio()函数对这些片段进一步处理 --- old
                            result = process_vad_audio(audio_vad[beg:end], sv, lang)  # todo: async
                            logger.debug(f"[process_vad_audio] {result}")
                            audio_vad = audio_vad[end:]  # 已经处理过的片段移除，保留未处理的部分
                            last_vad_beg = last_vad_end = -1  # 重置 VAD 片段标记

                            if result is not None:
                                response = TranscriptionResponse(
                                    code=0,
                                    msg=f"success",
                                    data=format_str_v3(result[0]['text']),
                                )
                                await websocket.send_json(response.model_dump())

                            # 调用process_vad_audio()函数对这些片段进一步处理 --- 20240917
                            # speaker_label, result = process_vad_audio(audio_vad[beg:end], sv, lang)  # todo: async
                            # # logger.debug(f"[process_vad_audio] {result[0]}")
                            # audio_vad = audio_vad[end:]  # 已经处理过的片段移除，保留未处理的部分
                            # last_vad_beg = last_vad_end = -1  # 重置 VAD 片段标记
                            #
                            # if result is not None:
                            #     response = TranscriptionResponse(
                            #         code=0,
                            #         msg=f"success",
                            #         data=speaker_label + ": " + format_str_v3(result[0]['text']),
                            #     )
                            #     await websocket.send_json(response.model_dump())



    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        await websocket.close()
    finally:
        audio_buffer = np.array([])
        audio_vad = np.array([])
        cache.clear()
        logger.info("Cleaned up resources after WebSocket disconnect")


@app.post("/predict-sentiment/")
async def predict_sentiment(request: Request):
    request_data = await request.json()
    text = request_data.get("text")
    if text:
        sentiment = text_sentiment_inference(text)
        return {"sentiment": sentiment}
    else:
        return {"error": "No text provided"}


final_score_list = []  # 存储所有的最终得分
# update at 20240915
@app.post("/audio-predict-sentiment/")
async def audio_predict_sentiment():
    wav_file_path = "./temp_chunk.wav"

    if not os.path.exists(wav_file_path):
        raise HTTPException(status_code=404, detail="File not found")
    try:
        logger.debug(f"Processing file {wav_file_path}")
        feature_test_instance = calc_feature_all(wav_file_path)
        test_instance = [feature_test_instance[key] for key in selected_feature_name if key in feature_test_instance]
        # last semester score
        final_score = calculate_final_score(test_instance)
        # this semester score - [-1,0,1]
        final_sentiment_3_new = CNN_Model_Predication_New(test_instance)

        # if final_sentiment_3_new is list，then pick the first one
        if isinstance(final_sentiment_3_new, list):
            final_sentiment_3_new = final_sentiment_3_new[0]

        combine_score = calculate_combine_score(test_instance, final_score, final_sentiment_3_new)

        sentiment_category= determine_sentiment_category(final_sentiment_3_new)

        # update final_score_list
        final_score_list.append(float(combine_score))

        response = {
            "final_score": combine_score,
            # "final_emotion_8": final_emotion_8[0],
            "final_sentiment_3": sentiment_category,
            # "probability": probability
        }
        return response
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

# 更新折线图
@app.post("/update-chart")
async def update_chart():
    try:
        accumulate_end_time = 0  # 用于累积的结束时间
        end_time_list = []  # 存储所有的结束时间

        # 遍历存储 VAD 处理数据的 timeline_data
        for entry, score in zip(timeline_data, final_score_list):  # 确保 final_score_list 和 timeline_data 对应
            # 取出 VAD 处理后的音频的结束时间
            audio_end_time = entry["end_time_relative"]

            # 将相对结束时间叠加到总结束时间
            accumulate_end_time += audio_end_time - entry.get("start_time_relative", 0)  # 保证只加这段语音的时长
            # 将累计结束时间加入列表，保留两位小数
            end_time_list.append(round(accumulate_end_time, 2))
            # 返回结果作为响应
        response = {
            "end_time": end_time_list,
            "final_score": final_score_list
        }
        return response

    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating chart: {str(e)}")


tm = TopicModel()
@app.post("/topic-model/")
async def topic_model(request: Request):
    try:
        # 从 HTTP 请求体中提取 JSON 数据
        request_data = await request.json()
        text = request_data.get("text")  # 提取传递的文本

        if text:
            # 调用 getTopics 方法处理文本
            topics = tm.getTopics(text)
            print(f"Extracted topics: {topics}")

            # 将结果打包成 JSON 格式返回给客户端
            return {"topics": topics}
        else:
            return {"error": "No text provided"}
    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FastAPI app with a specified port.")
    parser.add_argument('--port', type=int, default=8000, help='Port number to run the FastAPI app on.')
    # parser.add_argument('--certfile', type=str, default='path_to_your_certfile', help='SSL certificate file')
    # parser.add_argument('--keyfile', type=str, default='path_to_your_keyfile', help='SSL key file')
    args = parser.parse_args()

    # uvicorn.run(app, host="0.0.0.0", port=args.port, ssl_certfile=args.certfile, ssl_keyfile=args.keyfile)
    uvicorn.run(app, host="127.0.0.1", port=args.port)
