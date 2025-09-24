import re
import logging
from typing import List

logger = logging.getLogger(__name__)

def simple_smart_split(text: str, max_tokens: int) -> List[str]:

    if not text.strip():
        return []

    # Simple approximation: 1 token ≈ 4 characters
    max_chars = max_tokens * 4
    
    # Split text into sentences at sentence boundaries, discarding the delimiters.
    sentences = re.split(r'[,.!?;:]\s*', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return []
    
    chunks = []
    current_chunk = ""
    
    sentence_queue = sentences[:]
    
    while sentence_queue:
        sentence = sentence_queue.pop(0)
        
        # 1. If current + next sentence < max chars, add sentence
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            if current_chunk:
                current_chunk += " "
            current_chunk += sentence
        else:
            # 2. Else, create a new chunk
            if current_chunk:
                chunks.append(current_chunk)
            
            # 2a. If next sentence len > max char, split it
            if len(sentence) > max_chars:
                words = sentence.split()
                temp_chunk = ""
                remaining_words = []
                
                # Add as many words as possible to the current line
                for i, word in enumerate(words):
                    if len(temp_chunk) + len(word) + 1 <= max_chars:
                        if temp_chunk:
                            temp_chunk += " "
                        temp_chunk += word
                    else:
                        remaining_words = words[i:]
                        break
                
                if temp_chunk:
                    chunks.append(temp_chunk)
                
                # The rest of the sentence becomes the new "next sentence"
                if remaining_words:
                    sentence_queue.insert(0, " ".join(remaining_words))
                current_chunk = ""
            else:
                # 2b. Then new chunk starts with the current sentence
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks

def batch_split(text: str, max_tokens: int, batch_size: int) -> List[List[str]]:
    chunks = simple_smart_split(text, max_tokens)
    if not chunks:
        return []

    # Group chunks into batches
    num_chunks = len(chunks)
    first_batch_size = num_chunks % batch_size
    if first_batch_size == 0 and num_chunks > 0:
        first_batch_size = batch_size

    batches = []
    if first_batch_size > 0:
        batches.append(chunks[:first_batch_size])
    
    remaining_chunks = chunks[first_batch_size:]
    for i in range(0, len(remaining_chunks), batch_size):
        batches.append(remaining_chunks[i:i + batch_size])
        
    return batches