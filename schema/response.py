# Define the response model
from pydantic import BaseModel


class TranscriptionResponse(BaseModel):
    """ Data model for transcription response via web socket message """
    code: int
    msg: str
    data: str
    type: str  # e.g., 'stt' (speech-to-text), 'sentiment', 'score'
    timestamp: str  # Include timestamp as an ISO format string
    speaker_label: str = ""  # Speaker label


class AnalysisResponse(BaseModel):
    """ Data model for analysis response """
    code: int = 0
    msg: str = 'success'
    data: str
    type: str  # e.g. 'text-sentiment', 'topic', 'audio-sentiment'
    timestamp: str  # Include timestamp as an ISO format string
