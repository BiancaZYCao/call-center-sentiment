""" Configuration file """
from pydantic_settings import BaseSettings
from pydantic import BaseModel, Field

class STTConfig(BaseSettings):
    """ Settings for VAD, ASR and speaker detection """
    sv_thr: float = Field(0.335, description="Speaker verification threshold")
    chunk_size_ms: int = Field(100, description="Chunk size in milliseconds")
    sample_rate: int = Field(16000, description="Sample rate in Hz")
    bit_depth: int = Field(16, description="Bit depth")
    channels: int = Field(1, description="Number of audio channels")