""" ASR & Speaker Recognition Module """
import os
from pydantic_settings import BaseSettings
from pydantic import BaseModel, Field
# ASR model import
import soundfile as sf
from funasr import AutoModel
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


class ASRConfig(BaseSettings):
    sv_thr: float = Field(0.335, description="Speaker verification threshold")
    chunk_size_ms: int = Field(100, description="Chunk size in milliseconds")
    sample_rate: int = Field(16000, description="Sample rate in Hz")
    bit_depth: int = Field(16, description="Bit depth")
    channels: int = Field(1, description="Number of audio channels")


asr_config = ASRConfig()

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

vad_model = AutoModel(
    model="fsmn-vad",
    model_revision="v2.0.4",
    disable_pbar=True,  # 禁用进度条显示，通常用于防止在非交互式环境中出现多余的输出。
    max_end_silence_time=380,  # 设置最大结束静音时间（单位：毫秒）。如果在检测过程中静音持续超过这个时间，模型可能会认为语音段结束。
    speech_noise_thres=0.8,  # 语音与噪声之间的阈值，用于区分语音和噪声。值越大，模型越倾向于认为音频是噪声。
    disable_update=True  # 禁用模型的自动更新功能，防止在处理过程中更新模型参数。
)

# insert agent speech sample for identification, files under /speaker
# Note: to make it more efficient, recommend to put only one agent
speaker_speech_files_for_recognize = [
    # "speaker/agent_tube.wav",
    # "speaker/agent_0003.wav",
    # "speaker/agent_0001.wav",
    # "speaker/agent_0007.wav",
    "speaker/agent_0022.wav",
    # "speaker/agent_0027.wav",
    # "speaker/agent_0028.wav",
    # "speaker/agent_0032.wav",
]


def recognize_speaker_init(files):
    """ load speaker speech files to recognize in the dialog """
    recognize_speakers = {}
    for f in files:
        data, sr = sf.read(f, dtype="float32")
        k, _ = os.path.splitext(os.path.basename(f))
        recognize_speakers[k] = {
            "data": data,
            "sr": sr,
        }
    return recognize_speakers


recognize_agents = recognize_speaker_init(speaker_speech_files_for_recognize)


def recognize_agent_speaker_after_vad(audio, sv=True, lang="en"):
    speaker_label = "Client"
    if not sv:
        return speaker_label, asr_pipeline(audio, language=lang.strip())

    for k, v in recognize_agents.items():
        res_sv = sv_pipeline([audio, v["data"]], thr=asr_config.sv_thr)
        print(f"[speaker check] {k}: {res_sv}")
        if res_sv["score"] >= asr_config.sv_thr:
            print(f"[speaker check identified] {k}: score at {res_sv['score']}")
            speaker_label = "Agent"
            break
    return speaker_label, asr_pipeline(audio, language=lang.strip())
