import platform
import yaml
from dataclasses import dataclass


@dataclass
class AMAEConfig:
    # paths
    cfg_path: str = None
    data_dir: str = None
    encoder_ckpt: str = None  # set None will not load checkpoint
    decoder_ckpt: str = None
    workspace: str = None

    # audio Fband config
    # audio_len: float = 10
    sample_rate: int = 32000
    frame_length: int = 25
    frame_shift: int = 10
    mel_bins: int = 64

    # training config
    pre_training: bool = True
    load_encoder: bool = False
    load_decoder: bool = False
    batch_size: int = 128
    learning_rate: float = 0.001  # 1e-4 also workable
    max_epoch: int = 200
    lr_scheduler: list = (0.02, 0.05, 0.1)
    lr_scheduler_epoch: list = (10, 20, 30)
    mask_ratio: float = 0.
    norm_pix_loss: bool = True
    restore_mask_only: bool = True  # grid

    # model config
    extra_downsample_ratio: int = 1
    num_classes: int = 0  # set 0 means ssl pre-train task

    embed_dim: int = 96
    patch_size: int = 4
    window_size: int = 8
    spec_size: int = 256
    latent_channel: int = 32
    absolute_position_embedding: bool = False

    EC_attn_norm: bool = True
    EC_patch_norm: bool = True
    EC_qkv_bias: bool = True
    EC_mlp_ratio: float = 4.
    EC_num_head: list = (4, 8, 16, 32)
    EC_depth: list = (2, 2, 6, 2)
    EC_drop_rate: float = 0.
    EC_attn_drop_rate: float = 0.
    EC_drop_path_rate: float = 0.1

    DC_num_head: list = (32, 16, 8, 4)
    DC_depth: list = (2, 6, 2, 2)
    DC_mlp_ratio: float = 4.
    DC_drop_rate: float = 0.
    DC_attn_drop_rate: float = 0.
    DC_drop_path_rate: float = 0.1
    DC_attn_norm: bool = True
    DC_patch_norm: bool = True
    DC_qkv_bias: bool = True

    def __init__(self, cfg_path=None):
        if cfg_path is not None:
            self.cfg_path = cfg_path
            self.parse_config()

    def parse_config(self):
        def _sys_compatible(config):
            sys_key = 'win' if platform.system() == 'Windows' else 'linux'
            compatible_keys = config['sys_compatible_keys']
            for k in compatible_keys:
                assert k in config.keys(), f'???key {k}'
                config[k] = config[k][sys_key]
            config.pop('sys_compatible_keys')

        with open(self.cfg_path, 'r') as f:
            yaml_cfg = yaml.safe_load(f)
            _sys_compatible(yaml_cfg)

        for name, value in yaml_cfg.items():
            assert hasattr(self, name), f'data class miss attr {name}'
            setattr(self, name, value)

