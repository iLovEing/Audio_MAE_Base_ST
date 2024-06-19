import os
import h5py
import torch
import librosa
import pandas as pd
from utils import AMAEConfig
import torchaudio.compliance.kaldi as kaldi
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    def __init__(self, cfg: AMAEConfig):
        super().__init__()
        df = pd.read_csv(cfg.data_csv, index_col=0)
        assert 'type' in df.columns and 'file' in df.columns
        self.wav_files = list(df[df['type'] == 'train']['file'])
        self.h5f = h5py.File(cfg.hdf5_file, 'a') if cfg.hdf5_file is not None else None

        self.sr = cfg.sample_rate
        self.mel_bins = cfg.mel_bins
        self.frame_length = cfg.frame_length
        self.frame_shift = cfg.frame_shift
        freq_ratio = cfg.spec_size // cfg.mel_bins
        self.target_frame = int(cfg.spec_size * freq_ratio * cfg.extra_downsample_ratio)

    def wav2fbank(self, wav_f):
        sgnl, _ = librosa.load(wav_f, sr=self.sr)
        fbank = kaldi.fbank(torch.Tensor(sgnl).unsqueeze(0), sample_frequency=self.sr, num_mel_bins=self.mel_bins,
                            frame_length=self.frame_length, frame_shift=self.frame_shift,
                            use_energy=False, htk_compat=True, window_type='hanning', dither=0.0)
        return fbank

    def __getitem__(self, idx):
        wav_f = self.wav_files[idx]
        wav_name = os.path.basename(wav_f)
        if self.h5f is not None:
            if wav_name in self.h5f:
                fbank = torch.tensor(self.h5f[wav_name][:])
            else:
                fbank = self.wav2fbank(wav_f)
                self.h5f[wav_name] = fbank.numpy()
        else:
            fbank = self.wav2fbank(wav_f)

        p = self.target_frame - fbank.shape[0]
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:self.target_frame, :]

        return fbank.unsqueeze(0)

    def __len__(self):
        return len(self.wav_files)

