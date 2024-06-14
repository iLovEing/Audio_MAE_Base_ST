import os
import copy
import bisect
import numpy as np
from tqdm import tqdm
import torch
import random
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import librosa

from utils import parse_config
from model import AutoEncoder


class_map = {
    "standby": -1,
    "normal": 0,
    "knock": 1,
    "slide": 2,
    "shock": 3,
    "others": 4,
}


class STPOCDataset(Dataset):
    def __init__(self, datas):
        self.dataset = datas
        random.shuffle(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]["waveform"], self.dataset[index]["target"]

    def __len__(self):
        return len(self.dataset)


class STPOCDataPre:
    def __init__(self, data_dir, sr=32000, target_len=30.5, repeat=None, random_cut=True, empty_prob=0., added_file=False):
        print(f'pre process data "{data_dir}" start.')
        print(f'random_cut: {random_cut}, empty_prob: {empty_prob}, added_file: {added_file}, repeat: {repeat}')

        self.root = data_dir
        self.sr = sr
        self.target_len = target_len
        self.repeat = repeat if repeat is not None else {}

        self.train_enhance_wav = []
        self.test_enhance_wav = []
        self.train_data = []
        self.test_data = []

        # enhance file
        self.generate_enhance_wav()

        # normal
        self.process_normal_data(empty_prob=empty_prob)

        # abnormal
        self.process_abnormal_data(random_cut, empty_prob)

        random.shuffle(self.train_data)
        random.shuffle(self.test_data)
        print(f'finish pre process data.')

    @property
    def train_set(self):
        return self.train_data

    @property
    def test_set(self):
        return self.test_data

    def summary(self):
        print(f'data summary:')
        for k, v in class_map.items():
            train_num = 0
            test_num = 0
            for audio in self.train_data:
                if audio["target"] == v:
                    train_num += 1
            for audio in self.test_data:
                if audio["target"] == v:
                    test_num += 1
            print(f"class {k}, train len {train_num}, test len {test_num}")

    def process_abnormal_data(self, random_cut=False, empty_prob=0.):
        train_root = os.path.join(self.root, 'train', 'abnormal')
        test_root = os.path.join(self.root, 'test', 'abnormal')

        train_data = []
        for abnormal in os.listdir(train_root):
            # complete audio, generate by cut
            complete_dir = os.path.join(train_root, abnormal, 'complete')
            for d in os.listdir(complete_dir):
                temp_data = self.generate_wav_by_cut(os.path.join(complete_dir, d), abnormal)
                train_data += temp_data

            # incomplete audio, generate by enhance
            cut_dir = os.path.join(train_root, abnormal, 'cut')
            for d in os.listdir(cut_dir):
                person_dir = os.path.join(cut_dir, d)
                if os.path.exists(os.path.join(person_dir, 'ori')):
                    temp_data = self.generate_wav_by_cut(os.path.join(person_dir, 'ori'), abnormal)
                    train_data += temp_data
                temp_data = self.generate_wav_by_enhance(person_dir, abnormal, empty_prob=empty_prob)
                train_data += temp_data

        test_data = []
        for abnormal in os.listdir(test_root):
            # complete audio, generate by cut
            complete_dir = os.path.join(test_root, abnormal, 'complete')
            for d in os.listdir(complete_dir):
                temp_data = self.generate_wav_by_cut(os.path.join(complete_dir, d), abnormal,
                                                     random_cut=random_cut)
                test_data += temp_data

            # incomplete audio, generate by enhance
            cut_dir = os.path.join(test_root, abnormal, 'cut')
            for d in os.listdir(cut_dir):
                temp_data = self.generate_wav_by_enhance(os.path.join(cut_dir, d), abnormal,
                                                         train_data=False, empty_prob=empty_prob)
                test_data += temp_data

        print(f'3. process abnormal data, train : {len(train_data)}, test: {len(test_data)}')
        self.train_data += train_data
        self.test_data += test_data

    def generate_wav_by_enhance(self, file_dir, target, train_data=True, empty_prob=0.):
        result = []
        repeat_num = self.repeat[target] if target in self.repeat.keys() else 1
        for f in os.listdir(file_dir):
            if not f.endswith("wav"):
                continue

            audio_file = os.path.join(file_dir, f)
            for _ in range(repeat_num):
                wav = self.enhance_file(audio_file, train_data, empty_prob)
                result.append({
                    "file": audio_file,
                    "target": class_map[target],
                    "waveform": wav
                })
        return result

    def enhance_file(self, file_p, train_data=True, empty_prob=0.):
        def get_enhance_wav(wav_len, is_train):
            file_idx = -1
            enhance_wavs = self.train_enhance_wav if is_train else self.test_enhance_wav
            while file_idx < 0 or enhance_wavs[file_idx].shape[0] < wav_len:
                file_idx = random.randint(0, len(enhance_wavs) - 1)
            start_idx = random.randint(0, enhance_wavs[file_idx].shape[0] - wav_len)
            return enhance_wavs[file_idx][start_idx: start_idx + wav_len]

        result = np.zeros((int(self.sr * self.target_len)), dtype=np.float32)
        signal, _ = librosa.load(file_p, sr=self.sr)
        assert signal.shape[0] < self.sr * self.target_len, f"{file_p}"

        pos_s = random.randint(0, result.shape[0] - signal.shape[0])
        pos_e = pos_s + signal.shape[0]
        result[pos_s: pos_e] = signal

        if not pos_s == 0 and random.random() >= empty_prob:
            result[: pos_s] = get_enhance_wav(pos_s, train_data)
        if not pos_e == result.shape[0] and random.random() >= empty_prob:
            result[pos_e:] = get_enhance_wav(result.shape[0] - pos_e, train_data)

        return result

    def process_normal_data(self, random_cut=False, empty_prob=0.):
        train_root = os.path.join(self.root, 'train', 'normal')
        test_root = os.path.join(self.root, 'test', 'normal')

        train_data = self.generate_wav_by_cut(train_root, 'normal')
        test_data = self.generate_wav_by_cut(test_root, 'normal')

        print(f'2. process normal data, train: {len(train_data)}, test: {len(test_data)}')
        self.train_data += train_data
        self.test_data += test_data

    def generate_wav_std(self, file_dir, target):
        results = []
        repeat_num = self.repeat[target] if len(self.repeat) > 0 and target in self.repeat.keys() else 1
        for f in os.listdir(file_dir):
            if not f.endswith('wav'):
                continue
            for _ in range(repeat_num):
                audio_file = os.path.join(file_dir, f)
                signal, _ = librosa.load(audio_file, sr=self.sr)
                results.append({
                    "file": audio_file,
                    "target": class_map[target],
                    "waveform": signal
                })
        return results

    # for audios longer than target
    def generate_wav_by_cut(self, file_dir, target, random_cut=False):
        results = []
        repeat_num = self.repeat[target] if len(self.repeat) > 0 and target in self.repeat.keys() else 1
        for f in os.listdir(file_dir):
            if not f.endswith('wav'):
                continue
            for _ in range(repeat_num):
                audio_file = os.path.join(file_dir, f)
                wavs = self._cut_audio(audio_file, random_cut)
                for wav in wavs:
                    results.append({
                        "file": audio_file,
                        "target": class_map[target],
                        "waveform": wav
                    })
        return results

    def _cut_audio(self, file_p, random_cut=False):
        result = []
        signal, _ = librosa.load(file_p, sr=self.sr)
        wav_len = signal.shape[0]
        split_len = int(self.target_len * self.sr)
        assert wav_len >= split_len

        split_num = np.ceil(wav_len / split_len)
        split_delta = (wav_len - split_len) / (split_num - 1)
        split_heads = [int(idx * split_delta) for idx in range(int(split_num))]

        for start in split_heads:
            result.append(np.array(signal[start: start + split_len]))

        if random_cut:
            start = random.randint(0, signal.shape[0] - split_len)
            result.append(np.array(signal[start: start + split_len]))

        return result

    def generate_enhance_wav(self):
        train_enhance_root = os.path.join(self.root, 'train', 'enhance')
        test_enhance_root = os.path.join(self.root, 'test', "enhance")
        train_enhance_files = [os.path.join(train_enhance_root, x) for x in os.listdir(train_enhance_root)]
        test_enhance_files = [os.path.join(test_enhance_root, x) for x in os.listdir(test_enhance_root)]

        for audio_file in train_enhance_files:
            signal, _ = librosa.load(audio_file, sr=self.sr)
            self.train_enhance_wav.append(signal)
        for audio_file in test_enhance_files:
            signal, _ = librosa.load(audio_file, sr=self.sr)
            self.test_enhance_wav.append(signal)

        print(f'1. generate enhance wavs, train: {len(self.train_enhance_wav)}, test: {len(self.test_enhance_wav)}')
        random.shuffle(self.train_enhance_wav)
        random.shuffle(self.test_enhance_wav)


def get_dataset(data_dir):
    poc_data = STPOCDataPre(data_dir, target_len=30.72, repeat={'shock': 2}, empty_prob=0.2)
    poc_data.summary()
    train_data = poc_data.train_set
    test_data = poc_data.test_set
    return train_data, test_data


def lr_foo(epoch):
    if epoch < 3:
        # warm up lr
        lr_scale = g_cfg["lr_scheduler"][epoch]
    else:
        # warmup schedule
        lr_pos = int(-1 - bisect.bisect_left(g_cfg["lr_scheduler_epoch"], epoch))
        if lr_pos < -3:
            lr_scale = max(g_cfg["lr_scheduler"][0] * (0.98 ** epoch), 0.03 )
        else:
            lr_scale = g_cfg["lr_scheduler"][lr_pos]
    return lr_scale


def train(cfg, train_data, test_data):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'training device: {device}')

    train_set = STPOCDataset(train_data)
    test_set = STPOCDataset(test_data)
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True)
    test_loader = DataLoader(test_set, batch_size=cfg["batch_size"], shuffle=True)

    ae_model = AutoEncoder(cfg)
    optimizer = torch.optim.AdamW(
        params=filter(lambda p: p.requires_grad, ae_model.parameters()),
        lr=cfg["learning_rate"],
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.05,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_foo
    )
    loss_f = nn.CrossEntropyLoss()

    stop_step = 0
    best_model = None
    record_val_acc = 0.0
    record_val_loss = np.inf
    record_train_acc = 0
    record_train_loss = np.inf
    max_epoch = cfg["max_epoch"]

    # stop_by = "train_loss"
    # stop_by = "val_loss"
    stop_by = "acc"
    early_stop = 8

    ae_model = ae_model.to(device)
    for epoch in range(max_epoch):
        print(
            f"***** epoch[{epoch + 1}/{max_epoch}] lr: {optimizer.state_dict()['param_groups'][0]['lr']} *****")
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        # training
        ae_model.train()
        for batch in tqdm(train_loader, desc="training"):
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = ae_model(features)
            loss = loss_f(outputs['result'], labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            _, train_pred = torch.max(outputs['result'], 1)  # get the index of the class with the highest probability
            train_acc += (train_pred.detach() == labels.detach()).sum().item()
            train_loss += loss.item() * labels.shape[0]
        train_acc = train_acc / len(train_set)
        train_loss = train_loss / len(train_set)

        ae_model.eval()  # set the model to evaluation mode
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="testing"):
                features, labels = batch
                features = features.to(device)
                labels = labels.to(device)
                outputs = ae_model(features)
                loss = loss_f(outputs['result'], labels)

                _, val_pred = torch.max(outputs['result'], 1)
                val_acc += (
                            val_pred.detach() == labels.detach()).sum().item()  # get the index of the class with the highest probability
                val_loss += loss.item() * labels.shape[0]
        val_acc = val_acc / len(test_set)
        val_loss = val_loss / len(test_set)
        print(
            f"[{epoch + 1}/{max_epoch}] early stop step {stop_step + 1}, Train Acc: {format(train_acc, '.4f')} Loss: {format(train_loss, '.4f')} | "
            f"Val Acc: {format(val_acc, '.4f')} loss: {format(val_loss, '.4f')}")

        if stop_by == "acc":
            refresh = val_acc > record_val_acc
        elif stop_by == "train_loss":
            refresh = train_loss < record_train_loss
        elif stop_by == "val_loss":
            refresh = val_loss < record_val_loss
        else:
            assert 0, "unkonwn reshresh type"

        if refresh:
            record_val_acc = val_acc
            record_val_loss = val_loss
            record_train_acc = train_acc
            record_train_loss = train_loss
            best_ckpt = copy.deepcopy(ae_model.state_dict())
            stop_step = 0
            print(f"refresh best model with val acc {format(record_val_acc, '.4f')}")
        else:
            stop_step += 1
            if stop_step >= early_stop:
                ae_model.load_state_dict(best_ckpt)
                ae_model.save(cfg["saved_dir"], 'test')
                final_str = (f"training stoped by {stop_by}, stop info:\n"
                             f"at epoch {epoch + 1}\n"
                             f"train acc: {format(record_train_acc, '.4f')}, train loss: {format(record_train_loss, '.4f')}\n"
                             f"val acc {format(record_val_acc, '.4f')}, val loss: {format(record_val_loss, '.4f')}\n")
                print(final_str)
                with open(os.path.join(cfg["saved_dir"], f"info_stopby_{stop_by}_v0.txt"), 'w') as f:
                    f.write(final_str)
                break


def main(cfg):
    train_data, test_data = get_dataset(cfg['data_dir'])
    train(cfg, train_data, test_data)


if __name__ == '__main__':
    g_cfg = parse_config()
    main(g_cfg)
