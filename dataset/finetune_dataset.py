import os
import h5py
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from utils import AMAEConfig

from .dataset_utils import kaldi_wav2fbank


class FinetuneAS2k(Dataset):
    def __init__(self, cfg: AMAEConfig, key='train'):
        super().__init__()
        df = pd.read_csv(cfg.data_csv, index_col=0)
        assert 'type' in df.columns and 'file' in df.columns and 'label' in df.columns
        df = df[df['type'] == key]
        self.wav_files = df['file'].tolist()
        self.labels = df['label'].tolist()

        self.use_hdf5 = True if cfg.hdf5_dir is not None else False
        hdf5_f = os.path.join(cfg.hdf5_dir, f'ft_{key}.hdf5') if self.use_hdf5 else None
        self.h5f = h5py.File(hdf5_f, mode='r', locking=False) if self.use_hdf5 else None

        self.num_classes = cfg.num_classes
        self.use_roll = cfg.roll_mag_aug
        self.sr = cfg.sample_rate
        self.mel_bins = cfg.mel_bins
        self.frame_length = cfg.frame_length
        self.frame_shift = cfg.frame_shift
        freq_ratio = cfg.spec_size // cfg.mel_bins
        self.target_frame = int(cfg.spec_size * freq_ratio * cfg.extra_downsample_ratio)

    def __getitem__(self, idx):
        # fbank
        wav_f = self.wav_files[idx]
        wav_name = os.path.basename(wav_f)
        fbank = torch.tensor(self.h5f[wav_name][:]) \
            if self.use_hdf5 and wav_name in self.h5f \
            else kaldi_wav2fbank(wav_f)

        p = self.target_frame - fbank.shape[0]
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:self.target_frame, :]

        # label
        label_indices = np.zeros(self.num_classes)
        for label_str in self.labels[idx].split(','):
            label_indices[int(label_str)] = 1.0

        return fbank.unsqueeze(0), torch.tensor(label_indices), wav_f

    def __len__(self):
        return len(self.wav_files)

