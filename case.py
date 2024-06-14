import torch
import os
import torch.nn as nn

from .model import (get_fband_model, get_wav2img_model, get_swin_transformer_encoder,
                    get_st_classifier, get_swin_transformer_decoder)


class TrainAutoEncoder:
    def __init__(self, config, load_ckpt=True):
        super().__init__()
        self.fband = get_fband_model(config, load_ckpt)
        self.wav2img = get_wav2img_model(config, load_ckpt)
        self.encoder = get_swin_transformer_encoder(config, load_ckpt)
        self.classifier = get_st_classifier(config, load_ckpt)
        self.decoder = get_swin_transformer_decoder(config, type='CNN')

    def save(self, _dir, suffix=''):
        torch.save(self.fband, os.path.join(_dir, f"fband_{suffix}.pth"))
        torch.save(self.wav2img, os.path.join(_dir, f"wav2img_{suffix}.pth"))
        torch.save(self.encoder, os.path.join(_dir, f"encoder_{suffix}.pth"))
        torch.save(self.classifier, os.path.join(_dir, f"classifier_{suffix}.pth"))

