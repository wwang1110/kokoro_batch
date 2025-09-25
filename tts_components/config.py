import os

class Config:
    def __init__(self):
        self.kokoro_fp16 = os.getenv("KOKORO_FP16", "false").lower() in ('true', '1', 't')
        self.kokoro_max_batch_size = int(os.getenv("KOKORO_MAX_BATCH_SIZE", "16"))
        self.kokoro_max_chars_per_chunk = int(os.getenv("KOKORO_MAX_CHARS_PER_CHUNK", "200"))
