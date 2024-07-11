import os
import h5py
import librosa
import bisect
import pandas as pd
import numpy as np
import torchaudio.compliance.kaldi as kaldi
import torch
from tqdm import tqdm


def generate_data_csv(csv_f, **kwargs):
    wav_df = pd.DataFrame(data=None, columns=["file", "type", "label"])
    if 'train' in kwargs.keys():
        for _f_info in kwargs['train']:
            wav_df.loc[len(wav_df.index)] = [_f_info['file'], 'train', _f_info['label']]

    if 'eval' in kwargs.keys():
        for _f_info in kwargs['eval']:
            wav_df.loc[len(wav_df.index)] = [_f_info['file'], 'eval', _f_info['label']]

    if 'test' in kwargs.keys():
        for _f_info in kwargs['test']:
            wav_df.loc[len(wav_df.index)] = [_f_info['file'], 'test', _f_info['label']]

    wav_df.to_csv(csv_f)


def generate_hdf5(wav_files, hdf5, file_apart=1, sr=32000, mel_bins=64, frame_length=25, frame_shift=10):
    print(f'handle wav: {len(wav_files)}, file_apart {file_apart}')

    # creat h5f
    if file_apart == 1:
        h5f = h5py.File(hdf5, 'a')
    else:
        suffix = hdf5.split('.')[-1]
        prefix = hdf5[:-len(suffix) - 1]
        hdf5_files = [prefix + f'_{i}.' + suffix for i in range(10)]
        h5fs = [h5py.File(_hdf5_f, mode='a') for _hdf5_f in hdf5_files]
        h5f_idx_map = [round(i * len(wav_files) / 10) for i in range(1, 1 + 10)]

    for idx, wav_file in tqdm(enumerate(wav_files)):
        # parse fbank
        wav_name = os.path.basename(wav_file)
        fbank = kaldi_wav2fbank(wav_name, sr=sr, mel_bins=mel_bins, frame_length=frame_length, frame_shift=frame_shift)
        # write h5f
        if file_apart == 1:
            h5f[wav_name] = fbank.numpy()
        else:
            h5f = h5fs[bisect.bisect_left(h5f_idx_map, idx)]
            h5f[wav_name] = fbank.numpy()

    if file_apart == 1:
        h5f.close()
    else:
        for h5f in h5fs:
            h5f.close()

    print(f'finish writing {len(wav_files)}')


def norm_waveform(waveform, use_roll=False):
    waveform = waveform - waveform.mean()
    if use_roll:
        idx = np.random.randint(len(waveform))
        rolled_waveform = np.roll(waveform, idx)
        mag = np.random.beta(10, 10) + 0.5
        waveform = rolled_waveform * mag

    return waveform


def kaldi_wav2fbank(wav_f, sr=32000, mel_bins=64, frame_length=25, frame_shift=10):
    waveform, _ = librosa.load(wav_f, sr=sr)
    waveform = norm_waveform(waveform)
    fbank = kaldi.fbank(torch.tensor(waveform).unsqueeze(0), sample_frequency=sr, num_mel_bins=mel_bins,
                        frame_length=frame_length, frame_shift=frame_shift,
                        use_energy=False, htk_compat=True, window_type='hanning', dither=0.0)
    return fbank
