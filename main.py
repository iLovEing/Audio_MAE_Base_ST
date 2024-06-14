import torch
import librosa

from model import get_hts_at_model
from utils import parse_config


if __name__ == '__main__':
    device = torch.device('cpu')

    hts_at_config = parse_config()
    hts_at_model = get_hts_at_model(hts_at_config, state_key='infer_model', strict=False)
    hts_at_model.eval()

    test_wav = r'datas\test.wav'
    signal, _ = librosa.load(test_wav, sr=32000, mono=True)
    model_input = torch.Tensor(signal).unsqueeze(0)
    with torch.no_grad():
        result = hts_at_model(model_input)


