""" Swin Transformer
A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`
    - https://arxiv.org/pdf/2103.14030

Code/weights from https://github.com/microsoft/Swin-Transformer

"""

import torch
import random
import torch.nn as nn
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

from utils import AMAEConfig

from .model_utils import init_weights, reshape_wav2img, reshape_img2wav
from .st_layers import PatchMerging, BasicLayer, PatchEmbed


class STEncoder(nn.Module):
    r""" HTS-AT:hierarchical token-semantic audio transformer based on Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        # audio pre-process args
            spec_size (int | tuple(int)): Input Spectrogram size. Default 256
            n_fft (int): stft window size. Default 1024
            mel_bins (int): mel-spectrogram size. Default 1024
            hop_size (int): stft hop_size. Default 320
            sample_rate (int): audio sample rate.
            f_min (int): min filter bank freq. Default 0
            f_max (int): max filter bank freq.
        # swin transformer args
            patch_size (int | tuple(int)): Patch size. Default: 4
            in_chans (int): Number of input image channels. Default: 1 (mono)
            abslt_pos_ebd (bool): If True, add absolute position embedding to the patch embedding. Default: False
            num_classes (int): Number of classes for classification head. Default: None
            embed_dim (int): Patch embedding dimension. Default: 96
            depths (tuple(int)): Depth of each HTSAT-Swin Transformer layer.
            num_heads (tuple(int)): Number of attention heads in different layers.
            window_size (int): Window size. Default: 8
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
            qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
            drop_rate (float): Dropout rate. Default: 0
            attn_drop_rate (float): Attention dropout rate. Default: 0
            drop_path_rate (float): Stochastic depth rate. Default: 0.1
            norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
            patch_norm (bool): If True, add normalization after patch embedding. Default: True

    """

    def __init__(self, cfg: AMAEConfig, in_chans=1):
        #  qk_scale = None,
        super().__init__()

        self.sr = cfg.sample_rate
        self.mel_bins = cfg.mel_bins

        self.ds_ratio = cfg.extra_downsample_ratio
        self.num_classes = cfg.num_classes
        self.freq_ratio = cfg.spec_size // cfg.mel_bins
        self.target_freq = cfg.spec_size // self.freq_ratio

        self.spec_size = cfg.spec_size
        self.patch_size = cfg.patch_size
        self.embed_dim = cfg.embed_dim
        self.window_size = cfg.window_size
        self.abslt_pos_ebd = cfg.absolute_position_embedding
        self.latent_ch = cfg.latent_channel
        self.mlp_ratio = cfg.EC_mlp_ratio
        self.depths = cfg.EC_depth
        self.num_heads = cfg.EC_num_head
        self.drop_rate = cfg.EC_drop_rate
        self.attn_drop_rate = cfg.EC_attn_drop_rate
        self.drop_path_rate = cfg.EC_drop_path_rate
        self.qkv_bias = cfg.EC_qkv_bias
        self.attn_norm = cfg.EC_attn_norm
        self.patch_norm = cfg.EC_patch_norm
        self.num_layers = len(self.depths)

        self.pre_training = cfg.pre_training
        self.mask_ratio = cfg.mask_ratio
        self.restore_mask_only = cfg.restore_mask_only

        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
                                               freq_drop_width=8, freq_stripes_num=2)  # 2 2
        self.bn0 = nn.BatchNorm2d(self.mel_bins)

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size, in_c=in_chans, embed_dim=self.embed_dim, extra_ds_ratio=self.ds_ratio,
            abslt_pos_ebd=self.abslt_pos_ebd, norm_layer=nn.LayerNorm if self.patch_norm else nn.Identity()
        )
        self.pos_drop = nn.Dropout(p=self.drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]

        # build layers
        self.layers = nn.ModuleList()

        for i_layer in range(self.num_layers):
            # 注意这里构建的stage和论文图中有些差异
            # 这里的stage不包含该stage的patch_merging层，包含的是下个stage的
            layers = BasicLayer(dim=int(self.embed_dim * 2 ** i_layer),
                                depth=self.depths[i_layer],
                                num_heads=self.num_heads[i_layer],
                                window_size=self.window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=self.qkv_bias,
                                drop=self.drop_rate,
                                attn_drop=self.attn_drop_rate,
                                drop_path=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                                norm_layer=nn.LayerNorm if self.attn_norm else nn.Identity(),
                                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None)
            self.layers.append(layers)

        # stage4输出特征矩阵的channels
        self.norm = nn.LayerNorm(int(self.embed_dim * 2 ** (self.num_layers - 1))) if self.attn_norm else nn.Identity()

        st_output_channel = int(self.embed_dim * 2 ** (self.num_layers - 1))
        st_output_size = self.spec_size // (2 ** (self.num_layers - 1)) // self.patch_size
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.latent_head = nn.Conv2d(
            in_channels=st_output_channel,
            out_channels=self.latent_ch,
            kernel_size=(1, 1)
        ) if self.latent_ch is not None else nn.Identity()

        self.tscam_conv = nn.Conv2d(
            in_channels=self.latent_ch if self.latent_ch is not None else st_output_channel,
            out_channels=self.num_classes,
            kernel_size=(3, st_output_size // self.freq_ratio),
            padding=(1, 0)
        ) if self.num_classes != 0 else nn.Identity()

        self.apply(init_weights)

    def standardization_audio(self, x):
        batch, channel, time, freq = x.shape
        assert freq <= self.target_freq, \
            f"the freq size({freq}) should less than or equal to the swin input size({self.target_freq})"

        if freq < self.target_freq:
            x = nn.functional.interpolate(x, (x.shape[2], self.target_freq), mode="bicubic", align_corners=True)

        return x

    def random_mask(self, x):
        B, C, T, F = x.shape
        L = T * F
        x = x.reshape(B, C, -1)

        len_keep = int(L * (1 - self.mask_ratio))
        noise = torch.rand(B, C, L, device=x.device)  # noise in [0, 1]
        ids_shuffle = torch.argsort(noise, dim=2)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=2)

        keep_mask = torch.zeros([B, C, L], device=x.device)
        keep_mask[:, :, :len_keep] = 1
        keep_mask = torch.gather(keep_mask, dim=2, index=ids_restore)
        x_masked = torch.mul(x, keep_mask)

        remove_mask = torch.ones([B, C, L], device=x.device)
        if self.restore_mask_only:
            remove_mask[:, :, :len_keep] = 0
            remove_mask = torch.gather(remove_mask, dim=2, index=ids_restore)

        return x_masked.reshape(B, C, T, F), remove_mask.reshape(B, C, T, F)

    # input: [B, 1, T, mel_bin](e.g. [4, 1, 256, 256])
    def forward(self, x):
        x = self.standardization_audio(x)
        ori_fbank = x.clone().detach()

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        mask = None
        if self.training:
            if self.pre_training:
                x, mask = self.random_mask(x)
            else:
                x = self.spec_augmenter(x)

        # -> [B, C, img_H(f), img_W(T)](e.g. [4, 1, 256, 256])
        x = reshape_wav2img(x, self.freq_ratio)

        # [B, C, img_H, img_W](e.g. [4, 1, 256, 256]) -> [B, iH*iW(L), C'](e.g. [4, 4096, 96])
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)

        # [B, L, C] -> [B, L/4, 2C] -> [B, L/16, 4C] -> [B, L/64, 8C]
        # (e.g. [4, 4096, 96] -> [4, 1024, 192]) -> [4, 256, 384] -> [4, 64, 768]
        for layer in self.layers:
            x, H, W = layer(x, H, W)
        x = self.norm(x)  # [B, L, C]

        B, _, C = x.shape
        # [B, L, C] -> [B, C, H, W](e.g. [4, 64, 768] -> [4, 768, 8, 8])
        x = x.permute(0, 2, 1).contiguous().reshape(B, C, H, W)
        x = self.latent_head(x)

        # latent_img
        latent_wav = reshape_img2wav(x, self.freq_ratio)
        latent_feature = self.avgpool(torch.flatten(latent_wav, 2)).squeeze(-1)
        classifier_output = self.tscam_conv(latent_wav)
        classifier_output = torch.flatten(classifier_output, 2)  # B, C, T
        classifier_output = self.avgpool(classifier_output).squeeze(-1)

        return {
            'latent': latent_wav,
            'feature': latent_feature,
            'classifier': classifier_output,
            'ori_fbank': ori_fbank,
            'mask': mask,
        }

