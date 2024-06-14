import torch
import torch.nn as nn
from .model_utils import init_weights, reshape_wav2img, reshape_img2wav
from utils import AMAEConfig
from .st_layers import BasicLayer, PatchSeparate, PatchEmbedInverse


class STDecoder(nn.Module):
    def __init__(self, cfg: AMAEConfig):
        super().__init__()

        self.embed_dim = cfg.embed_dim
        self.patch_size = cfg.patch_size
        self.window_size = cfg.window_size
        self.st_dim = self.embed_dim * 2 ** (len(cfg.EC_depth) - 1)
        self.freq_ratio = cfg.spec_size // cfg.mel_bins
        self.ds_ratio = cfg.extra_downsample_ratio

        self.mlp_ratio = cfg.DC_mlp_ratio
        self.depths = cfg.DC_depth
        self.num_heads = cfg.DC_num_head
        self.drop_rate = cfg.DC_drop_rate
        self.attn_drop_rate = cfg.DC_attn_drop_rate
        self.drop_path_rate = cfg.DC_drop_path_rate
        self.qkv_bias = cfg.DC_qkv_bias
        self.attn_norm = cfg.DC_attn_norm
        self.patch_norm = cfg.DC_patch_norm
        self.num_layers = len(self.depths)
        self.latent_ch = cfg.latent_channel

        self.norm_pix_loss = cfg.norm_pix_loss

        self.latent_head_inverse = nn.Conv2d(
            in_channels=self.latent_ch,
            out_channels=self.st_dim,
            kernel_size=(1, 1)
        ) if self.latent_ch is not None else nn.Identity()
        self.ln0 = nn.LayerNorm(self.st_dim)

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]
        dpr.reverse()

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # 注意这里构建的stage和论文图中有些差异
            # 这里的stage不包含该stage的patch_merging层，包含的是下个stage的
            layers = BasicLayer(dim=int(self.st_dim / (2 ** i_layer)),
                                depth=self.depths[i_layer],
                                num_heads=self.num_heads[i_layer],
                                window_size=self.window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=self.qkv_bias,
                                drop=self.drop_rate,
                                attn_drop=self.attn_drop_rate,
                                drop_path=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                                norm_layer=nn.LayerNorm if self.attn_norm else nn.Identity(),
                                upsample=PatchSeparate if (i_layer < self.num_layers - 1) else None)
            self.layers.append(layers)

        self.pos_drop = nn.Dropout(p=self.drop_rate)
        self.patch_embed_inverse = PatchEmbedInverse(
            patch_size=self.patch_size, embed_dim=self.embed_dim, extra_ds_ratio=self.ds_ratio,
            norm_layer=nn.LayerNorm if self.patch_norm else nn.Identity()
        )

        self.apply(init_weights)

    def decoder_loss(self, target, mask, pred):
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        # loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches

        return loss

    def forward(self, latent_wav):
        x = reshape_wav2img(latent_wav, self.freq_ratio)

        x = self.latent_head_inverse(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.ln0(x)

        for layer in self.layers:
            x, H, W = layer(x, H, W)

        x = self.pos_drop(x)

        x = self.patch_embed_inverse(x, H, W)
        restored_x = reshape_img2wav(x, self.freq_ratio)

        return restored_x

