from .istftnet import Decoder
from .modules import CustomAlbert, ProsodyPredictor, TextEncoder, LSTMForward
import logging
from transformers import AlbertConfig
from typing import Dict, Union
import math
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KModel(torch.nn.Module):

    def __init__(
        self,
        config: Union[Dict, str, None],
        model: str,
        disable_complex: bool = False,
        fp16: bool = False
    ):
        super().__init__()

        self.fp16 = fp16
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")

        self.bert = CustomAlbert(AlbertConfig(vocab_size=config['n_token'], **config['plbert']))
        self.bert_encoder = torch.nn.Linear(self.bert.config.hidden_size, config['hidden_dim'])
        self.context_length = self.bert.config.max_position_embeddings
        self.predictor = ProsodyPredictor(
            style_dim=config['style_dim'], d_hid=config['hidden_dim'],
            nlayers=config['n_layer'], max_dur=config['max_dur'], dropout=config['dropout']
        )
        self.text_encoder = TextEncoder(
            channels=config['hidden_dim'], kernel_size=config['text_encoder_kernel_size'],
            depth=config['n_layer'], n_symbols=config['n_token']
        )
        self.decoder = Decoder(
            dim_in=config['hidden_dim'], style_dim=config['style_dim'],
            dim_out=config['n_mels'], disable_complex=disable_complex, **config['istftnet']
        )

        for key, state_dict in torch.load(model, map_location='cpu', weights_only=True).items():
            assert hasattr(self, key), key
            try:
                getattr(self, key).load_state_dict(state_dict)
            except:
                logger.debug(f"Did not load {key} from state_dict")
                state_dict = {k[7:]: v for k, v in state_dict.items()}
                getattr(self, key).load_state_dict(state_dict, strict=False)

        self.upsample_scale=math.prod(config['istftnet']['upsample_rates']) * config['istftnet']['gen_istft_hop_size']

        self.to(self.device)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        input_lengths: torch.LongTensor,
        voices: torch.FloatTensor,
        speeds: torch.FloatTensor
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        
        with torch.autocast(device_type=self.device, enabled=self.fp16):
            # Select reference style vector (ref_s) for each sequence in the batch based on input_lengths.
            # ref_s[:, :128] (First 128): Timbre / Voice Identity. Determines who is speaking.
            # ref_s[:, 128:] (The rest): Prosody / Style. Determines how they are speaking.
            ref_s = voices[torch.arange(input_ids.shape[0]), input_lengths - 1].squeeze(1)
            s = ref_s[:, 128:]

            text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(input_lengths.shape[0], -1).type_as(input_lengths)
            text_mask = torch.gt(text_mask+1, input_lengths.unsqueeze(1)).to(self.device)
            attention_mask = (~text_mask).float()

            # PLBERT is primarily used for extracting semantic and linguistic features from input text,
            bert_dur = self.bert(input_ids, attention_mask=attention_mask)
            d_en = self.bert_encoder(bert_dur).transpose(-1, -2)

            # This step likely combines the text and style information.
            d = self.predictor.text_encoder(d_en, s, input_lengths, text_mask)

            # This captures sequential dependencies in the combined text-style representation.
            x, _ = LSTMForward(self.predictor.lstm, d, input_lengths)

            # Predict the duration of each input token.
            duration = self.predictor.duration_proj(x)
            pred_dur = torch.round(((torch.sigmoid(duration)).sum(dim=-1).clamp(min=1) * attention_mask) / speeds.unsqueeze(1).to(self.device)).long() # b x seq_len
            seq_lengths = pred_dur.sum(axis=-1)
            max_frames = seq_lengths.max().item()

            # Creates an alignment matrix (pred_aln_trgs).
            frame_indices = torch.arange(max_frames, device=self.device).view(1,1,-1)
            duration_cumsum = pred_dur.cumsum(dim=1).unsqueeze(-1)
            mask1 = duration_cumsum > frame_indices
            mask2 = frame_indices >= torch.cat([torch.zeros(duration.shape[0],1, 1, device=self.device), duration_cumsum[:,:-1,:]],dim=1) # b x seq_len x max_dur
            pred_aln_trgs = (mask1 & mask2).float()

            # Expand d and input_ids to the frame level using the alignment matrix.
            en = d.transpose(-1, -2) @ pred_aln_trgs
            t_en = self.text_encoder(input_ids, input_lengths, text_mask)
            asr = t_en @ pred_aln_trgs

            updated_frame_mask = (frame_indices.squeeze(1).expand(en.shape[0], -1) >= seq_lengths.unsqueeze(1)).to(self.device)
            updated_frame_mask = (~updated_frame_mask).float()

            # Predict the fundamental frequency (F0) and noise components for each frame
            F0_pred, N_pred = self.predictor.F0Ntrain(en , s, seq_lengths, updated_frame_mask)

            # Generate a mel-spectrogram similar intermediate representation (b, 512, l), will be upsampled to (b, 128, l*upsample_scale)
            m = self.decoder(asr, F0_pred, N_pred, ref_s[:, :128], updated_frame_mask)

            # Converts the intermediate representation(mel-spectrogram) into the final raw audio waveform.
            updated_frame_mask = F.interpolate(updated_frame_mask.unsqueeze(1), scale_factor=2, mode='nearest').squeeze(1)
            audio = self.decoder.generator(m, ref_s[:, :128], F0_pred.squeeze(1), updated_frame_mask).squeeze(1)
            audio = (audio * 32767).to(torch.int16)
            frame_lengths = (seq_lengths * (audio.shape[-1]//max_frames)).long()

            return audio, frame_lengths
    