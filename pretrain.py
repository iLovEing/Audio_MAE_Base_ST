import os
import sys
import argparse
from datetime import datetime
from tqdm import tqdm
from loguru import logger
import bisect
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist

from model import STEncoder, STDecoder
from utils import AMAEConfig
from dataset import PretrainDataset


def prepare_environment(cfg):
    # paths
    assert cfg.workspace is not None, f'None workspace dir'
    time_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    cfg.workspace = os.path.join(cfg.workspace, time_str)
    os.makedirs(cfg.workspace, exist_ok=True)

    # log
    # logger.remove()
    # logger.add(sys.stdout,
    #            format="{time:HH:mm:ss} | <red>{level}</red> | <green>{file}</green> | <b><i>{message}</i></b>")
    logger.add(
        os.path.join(cfg.workspace, 'log.log'),
        enqueue=True,
        encoding="utf-8",
        level="INFO"
    )

    logger.info(f'########## train start at {time_str}########## train start at ')
    logger.info(f'config: {cfg.cfg_path}')
    logger.info(f'workspace: {cfg.workspace}')


# ckpt: https://www.cnblogs.com/booturbo/p/17358917.html
def train(cfg: AMAEConfig, ddp=False):
    def _lr_foo(_epoch):
        if _epoch < 3:
            # warm up lr
            lr_scale = cfg.lr_scheduler[_epoch]
        else:
            # warmup schedule
            lr_pos = int(-1 - bisect.bisect_left(cfg.lr_scheduler_epoch, _epoch))
            if lr_pos < -3:
                lr_scale = max(cfg.lr_scheduler[0] * (0.98 ** _epoch), 0.03)
            else:
                lr_scale = cfg.lr_scheduler[lr_pos]
        return lr_scale

    if ddp:
        gpu = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )

        if gpu == 0:
            prepare_environment(cfg)
            logger.debug(f'enable DDP, world size: {world_size}')
        torch.cuda.set_device(gpu)
        logger.debug(f'pid: {os.getpid()}, ppid: {os.getppid()}, gpu: {gpu}-{rank}')
    else:
        prepare_environment(cfg)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        logger.debug(f'disable DDP, training device: {device}')

    # dataset
    dataset = PretrainDataset(root_dirs=[cfg.data_dir], audio_len=cfg.audio_len)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    if ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank
        )
        dataloader = DataLoader(dataset,
                                batch_size=cfg.batch_size,
                                shuffle=False,  # sampler will do shuffle
                                sampler=train_sampler)

    # model
    model_E = STEncoder(cfg)
    model_D = STDecoder(cfg)
    if ddp:
        model_E = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_E)
        model_E = nn.parallel.DistributedDataParallel(model_E, device_ids=[gpu])
        model_D = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_D)
        model_D = nn.parallel.DistributedDataParallel(model_D, device_ids=[gpu])
    else:
        model_E = model_E.to(device)
        model_D = model_D.to(device)

    opt_E = torch.optim.AdamW(
        params=[{'params': model_E.parameters(), 'initial_lr': cfg.learning_rate}],
        lr=cfg.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.05,
    )
    sched_E = torch.optim.lr_scheduler.LambdaLR(
        opt_E,
        lr_lambda=_lr_foo,
        last_epoch=-1,  # todo
    )

    opt_D = torch.optim.AdamW(
        params=[{'params': model_D.parameters(), 'initial_lr': cfg.learning_rate}],
        lr=cfg.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.05,
    )
    sched_D = torch.optim.lr_scheduler.LambdaLR(
        opt_D,
        lr_lambda=_lr_foo,
        last_epoch=-1,  # todo
    )

    step = 0
    loss_record = []
    model_E.train()
    model_D.train()
    for epoch in range(cfg.max_epoch):
        if ddp:
            train_sampler.set_epoch(epoch)

        progress_bar = dataloader if ddp and gpu != 0 else tqdm(dataloader, desc=f'Epoch {epoch + 1}/{cfg.max_epoch}')
        for batch in progress_bar:
            opt_E.zero_grad()
            opt_D.zero_grad()

            batch = batch.cuda() if ddp else batch.to(device)
            encoder_output = model_E(batch)
            decoder_output = model_D(encoder_output['latent'])
            loss = model_D.decoder_loss(
                target=encoder_output['ori_fband'],
                mask=encoder_output['mask'],
                pred=decoder_output
            )

            loss.backward()
            opt_E.step()
            opt_D.step()
            if (not ddp or gpu == 0) and step % 10 == 0:
                progress_bar.set_postfix(loss=loss.item())
                if step % 50 == 0:
                    logger.info(f'step {step} training loss {loss}.')
            loss_record.append(loss.item())
            step += 1
        sched_E.step()
        sched_D.step()

        ckpt_E = {'parameter': model_E.state_dict(),
                  'optimizer': opt_E.state_dict(),
                  'scheduler': sched_E.state_dict(),
                  'epoch': epoch,
                  }
        ckpt_D = {'parameter': model_D.state_dict(),
                  'optimizer': opt_D.state_dict(),
                  'scheduler': sched_D.state_dict(),
                  'epoch': epoch,
                  }
        torch.save(ckpt_E, os.path.join(cfg.workspace,
                                        f'encoder_ratio_{cfg.extra_downsample_ratio}_epoch_{epoch+1}.pth'))
        torch.save(ckpt_D, os.path.join(cfg.workspace,
                                        f'decoder_ratio_{cfg.extra_downsample_ratio}_epoch_{epoch+1}.pth'))

        with open(os.path.join(cfg.workspace, f'loss_record_epoch_{epoch+1}.txt'), 'w') as f:
            for loss in loss_record:
                f.write(f'{loss}\n')
        loss_record.clear()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', default=None, type=str)
    args = parser.parse_args()
    assert args.cfg_path is not None and os.path.exists(args.cfg_path), \
        f'config file does not exist: {args["cfg_path"]}'

    cfg = AMAEConfig(args.cfg_path)
    ddp = True if 'WORLD_SIZE' in os.environ.keys() else False
    train(cfg, ddp)


# ddp: torchrun --nnodes=1 --node_rank=0 --nproc_per_node=2 --master_addr="192.168.1.140" --master_port=23456 \
# pretrain.py --cfg_path E:\windows\project\python\Audio_MAE_Base_ST\config\pretrain.yaml
# normal: python pretrain.py --cfg_path E:\windows\project\python\Audio_MAE_Base_ST\config\pretrain.yaml
if __name__ == '__main__':
    main()
