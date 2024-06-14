import os
import torch
import librosa
import torch.nn.functional as F
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    def __init__(self, root_dirs: list, audio_len: float, sample_rate=32000):
        self.sr = sample_rate
        self.std_len = int(audio_len * self.sr)
        self.wav_files = []
        for _dir in root_dirs:
            for wav in os.listdir(_dir):
                wav_path = os.path.join(_dir, wav)
                self.wav_files.append(wav_path)

    def __getitem__(self, idx):
        wav_p = self.wav_files[idx]
        sgnl, _ = librosa.load(wav_p, sr=self.sr)

        if len(sgnl) < self.std_len:
            return torch.nn.functional.pad(torch.tensor(sgnl), (0, self.std_len-len(sgnl)), mode='constant', value=0)
        else:
            return torch.tensor(sgnl[:self.std_len])

    def __len__(self):
        return len(self.wav_files)