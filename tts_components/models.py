from pydantic import BaseModel, Field
from typing import List

class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to convert to speech")
    voice: str = Field(default="af_heart", description="Voice name")
    language: str = Field(default="en-US", description="Language code")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Speech speed")
    format: str = Field(default="wav", description="Audio format (wav/pcm)")

class BatchTTSRequest(BaseModel):
    requests: List[TTSRequest] = Field(..., description="List of TTS requests")
