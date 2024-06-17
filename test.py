from utils import AMAEConfig
from model import AudioMAE
import librosa
import os
import torch


def test():
    test_cfg = AMAEConfig(cfg_path=r'E:\windows\project\python\Audio_MAE_Base_ST\config\pretrain.yaml')
    test_m = AudioMAE(test_cfg)
    test_f = os.path.join('datas', 'test', 'test_10s.wav')
    signal, _ = librosa.load(test_f, sr=32000, mono=True)
    model_input = torch.Tensor(signal).unsqueeze(0)
    result = test_m(model_input)
    # print(test_m)


if __name__ == '__main__':
    test()

