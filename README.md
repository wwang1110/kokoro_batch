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