import torch
import torch.nn as nn
import sys

from utils import AMAEConfig
from .encoder import STEncoder
from .decoder import STDecoder


# audio masked auto encoder
class AudioMAE(nn.Module):
    def __init__(self, cfg: AMAEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = STEncoder(cfg)
        self.decoder = STDecoder(cfg)
        pass

    def forward(self, input):
        encode_info = self.encoder(input)
        restore = self.decoder(encode_info['latent'])

        return restore

