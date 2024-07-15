import os
import h5py
import torch
import librosa
import pandas as pd
import numpy as np
import bisect
import torchaudio.compliance.kaldi as kaldi
from torch.utils.data import Dataset
from utils import AMAEConfig


class ASPretrain(Dataset):
    def __init__(self, cfg: AMAEConfig):
        super().__init__()
        df = pd.read_csv(cfg.data_csv, index_col=0)
        assert 'type' in df.columns and 'file' in df.columns
        self.wav_files = list(df[df['type'] == 'train']['file'])

        self.use_hdf5 = True if cfg.hdf5_dir is not None else False
        self.h5f_l = self.get_hdf5_handler()
        self.h5f_idx_map = [round(i * len(self.wav_files) / 10) for i in range(1, 1+10)]

        self.use_roll = cfg.roll_mag_aug
        self.sr = cfg.sample_rate
        self.mel_bins = cfg.mel_bins
        self.frame_length = cfg.frame_length
        self.frame_shift = cfg.frame_shift
        freq_ratio = cfg.spec_size // cfg.mel_bins
        self.target_frame = int(cfg.spec_size * freq_ratio * cfg.extra_downsample_ratio)

    def get_hdf5_handler(self, h5_dir=None):
        if not self.use_hdf5:
            return []

        base_name = r'as_pretrain.hdf5'
        suffix = base_name.split('.')[-1]
        prefix = base_name[:-len(suffix) - 1]
        hdf5_files = [prefix + f'_{i}.' + suffix for i in range(10)]
        h5f_l = [h5py.File(os.path.join(h5_dir, _hdf5_f), mode='r', locking=False) for _hdf5_f in hdf5_files]
        return h5f_l

    def _pre_process(self, waveform):
        waveform = waveform - waveform.mean()
        if self.use_roll:
            idx = np.random.randint(len(waveform))
            rolled_waveform = np.roll(waveform, idx)
            mag = np.random.beta(10, 10) + 0.5
            waveform = rolled_waveform * mag

        return torch.Tensor(waveform)

    def _wav2fbank(self, wav_f):
        waveform, _ = librosa.load(wav_f, sr=self.sr)
        waveform = self._pre_process(waveform)
        fbank = kaldi.fbank(waveform.unsqueeze(0), sample_frequency=self.sr, num_mel_bins=self.mel_bins,
                            frame_length=self.frame_length, frame_shift=self.frame_shift,
                            use_energy=False, htk_compat=True, window_type='hanning', dither=0.0)
        return fbank

    def __getitem__(self, idx):
        wav_f = self.wav_files[idx]
        wav_name = os.path.basename(wav_f)
        if self.use_hdf5:
            h5f = self.h5f_l[bisect.bisect_left(self.h5f_idx_map, idx)]
            if wav_name in h5f:
                fbank = torch.tensor(h5f[wav_name][:])
            else:
                fbank = self._wav2fbank(wav_f)
        else:
            fbank = self._wav2fbank(wav_f)

        p = self.target_frame - fbank.shape[0]
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:self.target_frame, :]

        return fbank.unsqueeze(0)

    def __len__(self):
        return len(self.wav_files)

