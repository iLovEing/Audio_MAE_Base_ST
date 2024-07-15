from utils import AMAEConfig
from model import AudioMAE
import librosa
import os
import torch
from dataset import kaldi_wav2fbank


def test():
    test_cfg = AMAEConfig(cfg_path=r'config\pretrain.yaml')
    test_m = AudioMAE(test_cfg)
    test_f = os.path.join('datas', 'test', 'test_10s.wav')
    fbank = kaldi_wav2fbank(test_f)
    m = torch.nn.ZeroPad2d((0, 0, 0, 1024-fbank.shape[0]))
    fbank = m(fbank)
    result = test_m(fbank.unsqueeze(0).unsqueeze(0))
    # print(test_m)


if __name__ == '__main__':
    test()

