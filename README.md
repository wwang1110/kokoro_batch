# Kokoro Batch Inference

This project is based on [hexgrad/kokoro](https://github.com/hexgrad/kokoro) and extends it to support batch inferences for text-to-speech processing.

## Code Summary

This project provides a text-to-speech (TTS) pipeline capable of processing multiple text inputs in a single batch. The core components are:

*   **`pipeline.py`**: Implements the main `TTSPipeline` class that orchestrates the conversion of text to speech. It handles text processing, phoneme conversion, and synthesizes audio using the acoustic model.
*   **`kokoro/`**: Contains the core TTS model architecture, including the main `Kokoro` model (`model.py`), STFT-related network components (`istftnet.py`), and other neural network modules (`modules.py`).
*   **`tts_components/`**: Includes supplementary modules for the TTS pipeline, such as configuration management (`config.py`), grapheme-to-phoneme conversion (`integrated_g2p.py`), and other TTS models and utilities.
*   **`demo.py`**: A demonstration script that shows how to use the `TTSPipeline` to synthesize speech from a list of input texts and save the output as WAV files.

## Quick Start

1.  **Create a virtual environment:**
    ```bash
    uv venv .venv --python 3.12
    ```

2.  **Install dependencies:**
    ```bash
    uv pip install -r requirements.txt
    ```

3.  **Run the demo:**
    ```bash
    python demo.py

## Performance

Test results with `max_token=50` and `batch_size=32` on `RTX4090`:

```
INFO:tts_components.integrated_g2p:Batch processed 30 items, 30 successful
INFO:root:Source generation duration: 0.0451s
INFO:root:STFT processing duration: 0.0692s
INFO:root:Upsample 0 duration: 0.0135s
INFO:root:Noise processing 0 duration: 0.0311s
INFO:root:Resblocks 0 duration: 0.0256s
INFO:root:Upsample 1 duration: 0.0119s
INFO:root:Noise processing 1 duration: 0.0187s
INFO:root:Resblocks 1 duration: 0.2484s
INFO:root:Final processing and inverse STFT duration: 1.1648s
INFO:root:Total forward pass duration: 1.6319s
INFO:__main__:Total audio duration: 276.15 seconds.
```