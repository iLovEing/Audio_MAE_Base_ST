import os
import argparse
from datetime import datetime
from tqdm import tqdm
from loguru import logger
import bisect
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
import numpy as np
import copy

from model import STEncoder
from utils import AMAEConfig, concat_all_gather, calculate_stats
from dataset import FinetuneAS2k


def prepare_environment(cfg: AMAEConfig):
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

    logger.info(f'########## finetune on AS20k training start at {time_str} ##########')
    logger.info(f'config: {cfg}')


def train(cfg: AMAEConfig, ddp=False, amp=False, num_workers=0):
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

        if not ddp or gpu == 0:
            logger.info(f'adjust lr scale {lr_scale} at epoch {_epoch+1}')
        return lr_scale

    # 1. prepare
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
            logger.debug(f'enable DDP, world size: {world_size}, use amp: {amp}, num_workers {num_workers}')
        torch.cuda.set_device(gpu)
        device = torch.device(f"cuda:{gpu}")
        logger.debug(f'pid: {os.getpid()}, ppid: {os.getppid()}, gpu: {gpu}-{rank}')
    else:
        prepare_environment(cfg)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        logger.debug(f'disable DDP, training device: {device}, use amp: {amp}, num_workers {num_workers}')

    assert not (amp and device == torch.device("cpu")), f'do not suggest use amp on cpu'

    # 2. load ckpt
    ckpt = torch.load(cfg.encoder_ckpt, map_location=device) if cfg.load_encoder else None

    # 3. dataset
    dataset_train = FinetuneAS2k(cfg, key='train')
    dataset_eval = FinetuneAS2k(cfg, key='eval')
    if ddp:
        sampler_train = torch.utils.data.distributed.DistributedSampler(
            dataset_train,
            num_replicas=world_size,
            rank=rank
        )
        dataloader_train = DataLoader(dataset_train,
                                batch_size=cfg.batch_size,
                                shuffle=False,  # sampler will do shuffle
                                sampler=sampler_train,
                                num_workers=num_workers)

        sampler_eval = torch.utils.data.distributed.DistributedSampler(
            dataset_eval,
            num_replicas=world_size,
            rank=rank
        )
        dataloader_eval = DataLoader(dataset_eval,
                                batch_size=cfg.batch_size,
                                shuffle=False,  # sampler will do shuffle
                                sampler=sampler_eval,
                                num_workers=num_workers)
    else:
        dataloader_train = DataLoader(dataset_train,
                                batch_size=cfg.batch_size,
                                shuffle=True,
                                num_workers=num_workers)
        dataloader_eval = DataLoader(dataset_eval,
                                batch_size=cfg.batch_size,
                                shuffle=True,
                                num_workers=num_workers)

    # 4. model & optimizer & loss
    model = STEncoder(cfg).to(device)
    model.load_state_dict(ckpt['parameter'], strict=False if cfg.from_pretrain else True) if ckpt is not None else None
    if ddp:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    optimizer = torch.optim.AdamW(
        params=[{'params': model.parameters(), 'initial_lr': cfg.learning_rate}],
        lr=cfg.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.05,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=_lr_foo,
        last_epoch=-1 if ckpt is None or cfg.from_pretrain else ckpt['epoch'],
    )
    if ckpt is not None and not cfg.from_pretrain:
        optimizer.load_state_dict(optimizer['optimizer'])
        scheduler.load_state_dict(scheduler['scheduler'])

    criterion = nn.BCEWithLogitsLoss()

    step = 0
    start_epoch = 0 if ckpt is None or cfg.from_pretrain else ckpt['epoch']+1
    loss_record = []
    best_map = 0
    patience_step = 0
    scaler = torch.cuda.amp.GradScaler() if amp else None
    for epoch in range(start_epoch, cfg.max_epoch):
        # 5. training
        model.train()
        sampler_train.set_epoch(epoch) if ddp else None
        progress_bar_train = dataloader_train \
            if ddp and gpu != 0 \
            else tqdm(dataloader_train, desc=f'Epoch {epoch + 1}/{cfg.max_epoch} train')
        for batch in progress_bar_train:
            optimizer.zero_grad()
            feature, label, _ = batch
            feature = feature.to(device)
            label = label.to(device)
            if amp:
                with torch.cuda.amp.autocast():
                    output = model(feature)
                    loss = criterion(output['classifier'], label)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            else:
                output = model(feature)
                pred = output['classifier']
                loss = criterion(pred, label)
                loss.backward()
                optimizer.step()

            if (not ddp or gpu == 0) and step % 10 == 0:
                progress_bar_train.set_postfix(loss=loss.item())
                if step % 50 == 0:
                    logger.info(f'training loss {format(loss.item(), ".5f")} at step {step}')
            loss_record.append(loss.item())
            step += 1

        scheduler.step()

        # 6.eval
        with (torch.no_grad()):
            model.eval()
            preds = []
            targets = []
            sampler_eval.set_epoch(epoch) if ddp else None
            progress_bar_eval = dataloader_eval \
                if ddp and gpu != 0 \
                else tqdm(dataloader_eval, desc=f'Epoch {epoch + 1}/{cfg.max_epoch} eval')
            for batch in progress_bar_eval:
                feature, label, _ = batch
                feature = feature.to(device)
                label = label.to(device)
                if amp:
                    with torch.cuda.amp.autocast():
                        output = model(feature)
                else:
                    output = model(feature)
                pred = output['classifier']
                if ddp:
                    pred = concat_all_gather(pred)
                    label = concat_all_gather(label)

                preds.append(pred)
                targets.append(label)

            if not ddp or gpu == 0:
                preds = torch.cat(preds)
                targets = torch.cat(targets)
                loss = criterion(preds, targets).cpu().numpy()
                stats = calculate_stats(preds.cpu().numpy(), targets.cpu().numpy())

                ap = [stat['AP'] for stat in stats]
                map = np.mean([stat['AP'] for stat in stats])
                logger.info(f'eval mAP {format(map, ".6f")}, loss {format(loss, ".5f")} at epoch {epoch+1}')
                if map > best_map:
                    patience_step = 0
                    best_map = map
                    best_aps = ap
                    best_loss = loss
                    best_ckpt = {'parameter': copy.deepcopy(model.module.state_dict() if ddp else model.state_dict()),
                                 'optimizer': copy.deepcopy(optimizer.state_dict()),
                                 'scheduler': copy.deepcopy(scheduler.state_dict()),
                                 'epoch': copy.deepcopy(epoch),
                                 }
                    logger.info(f'refresh best model at epoch {epoch + 1}')
                else:
                    patience_step += 1
                if epoch + 1 == cfg.max_epoch or patience_step == cfg.patience:
                    logger.info(f'stop training at epoch {epoch + 1}, patience_step {patience_step}')
                    file_name = f'E_{epoch+1}_map_{format(best_map, ".6f")}_loss_{format(best_loss, ".5f")}'
                    torch.save(best_ckpt, os.path.join(cfg.workspace, f"{file_name}.pth"))
                    with open(os.path.join(cfg.workspace, f"{file_name}_ap.txt"), 'w') as f:
                        for _ap in best_aps:
                            f.write(f'{format(_ap, ".6f")}\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', default=None, type=str)
    parser.add_argument('--amp', action='store_true', help='use amp')
    parser.add_argument('--thread', default=0, type=int, help='num_workers')
    args = parser.parse_args()
    assert args.cfg_path is not None and os.path.exists(args.cfg_path), \
        f'config file does not exist: {args["cfg_path"]}'

    cfg = AMAEConfig(args.cfg_path)
    ddp = True if 'WORLD_SIZE' in os.environ.keys() else False
    amp = args.amp
    num_workers = args.thread
    train(cfg, ddp, amp, num_workers)


def debug_main():
    cfg_path = r'config\finetune_AS20k.yaml'
    cfg = AMAEConfig(cfg_path)
    ddp = False
    amp = True
    train(cfg, ddp, amp)


# ddp: torchrun --nnodes=1 --node_rank=0 --nproc_per_node=2 --master_addr="192.168.1.250" --master_port=23456 \
# pretrain.py --cfg_path /home/tlzn/users/zlqiu/project/Audio_MAE_Base_ST/config/pretrain.yaml --amp --thread 10
# normal: python pretrain.py --cfg_path /home/tlzn/users/zlqiu/project/Audio_MAE_Base_ST/config/finetune_AS20k.yaml
if __name__ == '__main__':
    # main()
    debug_main()
