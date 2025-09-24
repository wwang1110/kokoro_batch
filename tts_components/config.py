import os

class Config:
    """Service configuration"""
    def __init__(self):
        self.kokoro_fp16 = os.getenv("KOKORO_FP16", "false").lower() in ('true', '1', 't')
        
        # Batching config (without dynamic_batching flag)
        self.kokoro_max_batch_size = int(os.getenv("KOKORO_MAX_BATCH_SIZE", "32"))
        self.kokoro_max_tokens_per_chunk = int(os.getenv("KOKORO_MAX_TOKENS_PER_CHUNK", "50"))
