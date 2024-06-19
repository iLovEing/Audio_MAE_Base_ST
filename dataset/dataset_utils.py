import os
import pandas as pd


def generate_data_csv(csv_f, **kwargs):
    wav_df = pd.DataFrame(data=None, columns=["file", "type", "label"])
    if 'train_files' in kwargs.keys():
        for _f in kwargs['train_files']:
            wav_df.loc[len(wav_df.index)] = [_f, 'train', -1]

    if 'val_files' in kwargs.keys():
        for _f in kwargs['val_files']:
            wav_df.loc[len(wav_df.index)] = [_f, 'val', -1]

    if 'test_files' in kwargs.keys():
        for _f in kwargs['test_files']:
            wav_df.loc[len(wav_df.index)] = [_f, 'test', -1]

    wav_df.to_csv(csv_f)

