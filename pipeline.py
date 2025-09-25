
from kokoro import KModel
from tts_components import IntegratedG2P, G2PItem, simple_smart_split
from huggingface_hub import hf_hub_download, list_repo_files
from typing import Optional, List
import torch
import logging
import json
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)

MODEL_NAMES = {
    'hexgrad/Kokoro-82M': 'kokoro-v1_0.pth',
    'hexgrad/Kokoro-82M-v1.1-zh': 'kokoro-v1_1-zh.pth',
}

voice_names = [
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_heart",
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_nova",
    "af_river",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_echo",
    "am_eric",
    "am_fenrir",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    "am_santa",

    "bf_alice",
    "bf_emma",
    "bf_isabella",
    "bf_lily",
    "bm_daniel",
    "bm_fable",
    "bm_george",
    "bm_lewis",

    "ef_dora",
    "em_alex",
    "em_santa",
]

class KPipeline:
    
    def __init__(
        self,
        cache_dir: Optional[str],
        disable_complex: bool = False,
        fp16: bool = False
    ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        config, kokoro, voices = self._preload_model('hexgrad/Kokoro-82M', cache_dir)
        self.voices = voices
        self.vocab = config['vocab']

        self.g2p = IntegratedG2P()
        self.model = KModel(config=config, model=kokoro, disable_complex=disable_complex, fp16=fp16).eval()

    @property
    def kmodel(self):
        return self.model

    def _preload_model(self, repo_id, cache_dir):

        logger.info(f"Loading config from repo: {repo_id}")
        config = hf_hub_download(repo_id=repo_id, filename='config.json', cache_dir=cache_dir)
        with open(config, 'r', encoding='utf-8') as r:
            config = json.load(r)
            logger.debug(f"Loaded config: {config}")

        logger.info(f"Loading model from repo: {repo_id}")
        kokoro = hf_hub_download(repo_id=repo_id, filename=MODEL_NAMES[repo_id], cache_dir=cache_dir)

        logger.info("Loading all available voices...")
        # Get a list of all files in the 'voices' directory of the repo
        repo_files = list_repo_files(repo_id=repo_id, repo_type='model')
        voice_files = [f for f in repo_files if f.startswith('voices/') and f.endswith('.pt')]
        
        voices = {}
        for voice_file in voice_files:
            voice_name = voice_file.split('/')[-1].replace('.pt', '')
            if voice_name not in voice_names:
                continue
            voice_path = hf_hub_download(repo_id=repo_id, filename=voice_file, cache_dir=cache_dir)
            voice_tensor = torch.load(voice_path, map_location=self.device, weights_only=True)
            voices[voice_name] = voice_tensor

        logger.info(f"✅ Preloaded {len(voices)} voices.")

        return config, kokoro, voices

    def text_to_phonemes(self, items: List[G2PItem]) -> List[str]:
        try:
            results = self.g2p.convert_batch(items)
            return [result.phonemes for result in results]
        except Exception as e:
            raise RuntimeError(f"Integrated G2P conversion failed: {e}")

    def simple_smart_split(self, text: str, max_tokens: int) -> List[str]:
        return simple_smart_split(text, max_tokens)

    def prepare_params(
        self,
        phonemes: list[str],
        voices: list[str],
        speeds: list[float]
    ):
        # Converts the text-based phonemes into numerical IDs that the model can process
        input_ids = [
            torch.LongTensor([0, *(self.vocab[p] for p in ph if self.vocab.get(p) is not None), 0]).to(self.device)
            for ph in phonemes
        ]

        # Calculates the length of each sequence in input_ids
        input_lengths = torch.LongTensor([t.shape[0] for t in input_ids])

        # Pads the sequences in input_ids to ensure they are of uniform length
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0).to(self.device)

        # Selects the specific reference style vector.
        voice_packs = torch.stack([self.voices[voice] for voice in voices])

        return input_ids, input_lengths, voice_packs, torch.FloatTensor(speeds)

    def from_phonemes(
        self, 
        phonemes: list[str], 
        voices: list[str],
        speeds: list[float]
    ) -> torch.FloatTensor:
        input_ids, input_lengths, voice_packs, speeds = self.prepare_params(phonemes, voices, speeds)
        max_len = input_lengths.max().item()
        if max_len > 510:
            raise ValueError(f'Input sequence too long: {max_len} > 510')

        output, lengths = self.model(input_ids, input_lengths, voice_packs, speeds)

        # Trims the padding from each audio clip in the batch to match the original lengths
        audio = [a[:l].cpu() for a, l in zip(output, lengths)]
        return audio
