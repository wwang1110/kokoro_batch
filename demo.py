import time
from pipeline import KPipeline
import soundfile as sf
from tts_components import Config, G2PItem
import logging
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Main demo function"""
    # Global configuration and pipeline
    config = Config()
    
    pipeline = KPipeline(cache_dir="./cache", fp16=config.kokoro_fp16)
    
    # Comprehensive test cases
    test_data = [
        {"text": "Hello World", "voice": "af_heart", "speed": 1.0, "language": "en-US"},
        {"text": "Climate change is a pressing global issue that requires immediate and concerted efforts from all nations to mitigate its long-term consequences.", "voice": "bf_emma", "speed": 0.95, "language": "en-US"},
        {"text": "The study of quantum mechanics reveals a fascinating and often counterintuitive world, where particles can exist in multiple states simultaneously.", "voice": "af_heart", "speed": 1.0, "language": "en-US"},
        {"text": "Exploring the depths of the ocean, scientists continue to discover new species and ecosystems that challenge our understanding of life on Earth.", "voice": "am_eric", "speed": 1.05, "language": "en-US"},
        {"text": "The history of human civilization is a rich tapestry of cultures, innovations, and conflicts that have shaped the world we live in today.", "voice": "af_jessica", "speed": 1.0, "language": "en-US"},
        {"text": "In the realm of astrophysics, the search for extraterrestrial life remains a captivating endeavor, driving the development of powerful new telescopes and observation techniques.", "voice": "am_adam", "speed": 0.9, "language": "en-US"},
        {"text": "The principles of economics provide a framework for understanding how societies allocate scarce resources to meet the needs and wants of their populations.", "voice": "bf_emma", "speed": 1.0, "language": "en-US"},
        {"text": "From the towering skyscrapers of modern cities to the ancient pyramids of Egypt, architecture reflects the values and aspirations of the societies that create it.", "voice": "af_heart", "speed": 1.1, "language": "en-US"},
        {"text": "The intricate workings of the human brain, with its billions of neurons and trillions of connections, represent one of the greatest frontiers of scientific research.", "voice": "am_eric", "speed": 0.95, "language": "en-US"},
        {"text": "The evolution of music throughout history has produced a diverse array of genres and styles, each with its own unique cultural significance and emotional impact.", "voice": "af_jessica", "speed": 1.0, "language": "en-US"},
        {"text": "The development of renewable energy technologies, such as solar and wind power, is crucial for transitioning to a sustainable and environmentally friendly energy system.", "voice": "am_adam", "speed": 1.0, "language": "en-US"},
        {"text": "The philosophical debate between free will and determinism has captivated thinkers for centuries, raising profound questions about the nature of human agency and responsibility.", "voice": "bf_emma", "speed": 0.9, "language": "en-US"},
        {"text": "The vast and interconnected world of the internet has revolutionized communication, commerce, and access to information on a global scale.", "voice": "af_heart", "speed": 1.0, "language": "en-US"},
        {"text": "The study of genetics has unlocked the secrets of heredity and disease, paving the way for new medical treatments and a deeper understanding of human biology.", "voice": "am_eric", "speed": 1.0, "language": "en-US"},
        {"text": "The art of storytelling, from ancient oral traditions to modern cinematic masterpieces, has the power to transport us to different worlds and illuminate the human condition.", "voice": "af_jessica", "speed": 1.05, "language": "en-US"},
        {"text": "The challenges of space exploration, including the harsh conditions of microgravity and radiation, require innovative engineering solutions and a resilient human spirit.", "voice": "am_adam", "speed": 0.95, "language": "en-US"},
        {"text": "The diversity of languages spoken around the world is a testament to the creativity and adaptability of the human mind, with each language offering a unique window into a different culture.", "voice": "bf_emma", "speed": 1.0, "language": "en-US"},
        {"text": "The pursuit of happiness is a fundamental human desire, and psychologists have identified various factors that contribute to a fulfilling and meaningful life.", "voice": "af_heart", "speed": 1.0, "language": "en-US"},
        {"text": "The impact of social media on society is a complex and multifaceted issue, with both positive and negative consequences for individuals and communities.", "voice": "am_eric", "speed": 0.9, "language": "en-US"},
        {"text": "The beauty and complexity of the natural world, from the smallest insects to the largest galaxies, inspire awe and wonder in those who take the time to observe it.", "voice": "af_jessica", "speed": 1.0, "language": "en-US"},
        {"text": "The ethical implications of artificial intelligence, including issues of bias, privacy, and accountability, require careful consideration as these technologies become more powerful.", "voice": "am_adam", "speed": 1.0, "language": "en-US"},
        {"text": "The history of scientific discovery is filled with stories of perseverance, collaboration, and serendipity, as researchers have pushed the boundaries of human knowledge.", "voice": "bf_emma", "speed": 1.0, "language": "en-US"},
        {"text": "The importance of education in fostering critical thinking, creativity, and civic engagement cannot be overstated in a rapidly changing and interconnected world.", "voice": "af_heart", "speed": 0.95, "language": "en-US"},
        {"text": "The global economy is a dynamic and interdependent system, where events in one part of the world can have far-reaching consequences for businesses and consumers everywhere.", "voice": "am_eric", "speed": 1.0, "language": "en-US"},
        {"text": "The power of literature to evoke empathy and understanding is a testament to the enduring relevance of storytelling in an increasingly digital age.", "voice": "af_jessica", "speed": 1.0, "language": "en-US"},
        {"text": "The challenges of cybersecurity are constantly evolving, as new threats and vulnerabilities emerge in the digital landscape, requiring a proactive and adaptive approach to security.", "voice": "am_adam", "speed": 1.05, "language": "en-US"},
        {"text": "The benefits of a healthy lifestyle, including regular exercise and a balanced diet, are well-documented and contribute to both physical and mental well-being.", "voice": "bf_emma", "speed": 1.0, "language": "en-US"},
        {"text": "The role of government in a modern society is a subject of ongoing debate, with different political ideologies offering competing visions of the ideal relationship between the state and its citizens.", "voice": "af_heart", "speed": 0.9, "language": "en-US"},
        {"text": "The exploration of the human psyche, from the conscious mind to the depths of the unconscious, has been a central theme in psychology, philosophy, and the arts.", "voice": "am_eric", "speed": 1.0, "language": "en-US"},
        {"text": "The future of work is likely to be shaped by automation, artificial intelligence, and the gig economy, requiring individuals and organizations to adapt to new ways of learning and collaborating.", "voice": "af_jessica", "speed": 1.0, "language": "en-US"}
    ]
    #test_data = test_data[:3]  # Limit to first 3 for quicker demo
    
    text_chunks = []
    logger.info("Splitting texts into chunks...")
    for i, item in enumerate(test_data):
        # Use batch_split to break text into optimal chunks
        text_chunks.extend(pipeline.simple_smart_split(
            item["text"],
            config.kokoro_max_tokens_per_chunk
        ))

    batches = []
    for i in range(0, len(text_chunks), config.kokoro_max_batch_size):
        batches.append(text_chunks[i:i + config.kokoro_max_batch_size])

    count = 0
    durs = []
    for batch in batches:
    
        # Convert all text chunks to G2PItem objects for processing
        g2p_items = [G2PItem(text=chunk, language=item["language"]) for chunk in batch]

        # Convert all text chunks in batch to phonemes using integrated G2P
        phonemes_list = pipeline.text_to_phonemes(g2p_items)

        start_time = time.time()
        audio_tensors = pipeline.from_phonemes(
            phonemes=phonemes_list,  
            voices=[item["voice"]] * len(phonemes_list),  
            speeds=[item["speed"]] * len(phonemes_list)
        )
        durs.append(time.time() - start_time)
        logger.info(f"Batch of {len(batch)} chunks processed in {time.time() - start_time:.2f} seconds.")
        
        for audio in audio_tensors:
            logger.info(f"Generated audio tensor with shape: {audio.shape}")
    
            # Define the output file path
            output_file = f"demo/demo_output_{count}.wav"
        
            # Write the concatenated audio to the file
            sf.write(output_file, audio.numpy(), 24000)

            count += 1
        break

    logger.info(f"Total inference time for {count} chunks: {sum(durs):.2f} seconds.")
    total_audio_dur = sum([len(sf.read(f"demo/demo_output_{i}.wav")[0]) / 24000 for i in range(count)])
    logger.info(f"Total audio duration: {total_audio_dur:.2f} seconds.")

if __name__ == "__main__":
    asyncio.run(main())