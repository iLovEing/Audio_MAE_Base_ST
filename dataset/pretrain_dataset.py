import os
import torch
import librosa
from utils import AMAEConfig
import torchaudio.compliance.kaldi as kaldi
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    def __init__(self, cfg: AMAEConfig):
        self.sr = cfg.sample_rate
        self.mel_bins = cfg.mel_bins
        self.frame_length = cfg.frame_length
        self.frame_shift = cfg.frame_shift
        freq_ratio = cfg.spec_size // cfg.mel_bins
        self.target_frame = int(cfg.spec_size * freq_ratio * cfg.extra_downsample_ratio)

        self.wav_files = []
        for wav in os.listdir(cfg.data_dir):
            if not wav.endswith('wav'):
                continue
            wav_path = os.path.join(cfg.data_dir, wav)
            self.wav_files.append(wav_path)

    def __getitem__(self, idx):
        wav_p = self.wav_files[idx]
        sgnl, _ = librosa.load(wav_p, sr=self.sr)

        fbank = kaldi.fbank(torch.Tensor(sgnl).unsqueeze(0), sample_frequency=self.sr, num_mel_bins=self.mel_bins,
                            frame_length=self.frame_length, frame_shift=self.frame_shift,
                            use_energy=False, htk_compat=True, window_type='hanning', dither=0.0)

        p = self.target_frame - fbank.shape[0]
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:self.target_frame, :]

        return fbank.unsqueeze(0)


    def __len__(self):
        return len(self.wav_files)

