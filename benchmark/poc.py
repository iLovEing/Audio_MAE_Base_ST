import os
import sys
import numpy as np
import librosa
import random
import torch
from tqdm import tqdm

sys.path.append('..')
from model import AutoEncoder
from utils import parse_config, cal_feature_center, cos_similarity


CUDA_DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
POC_DATA_PATH = r'E:\project\python\poc\datas\ori'
POC_CLASS_MAP = {
    "standby": -1,
    "normal": 0,
    "knock": 1,
    "slide": 2,
    "shock": 3,
    "others": 4,
}
POC_CLASS_DICT = {
    -1: "standby",
    0: "normal",
    1: "knock",
    2: "slide",
    3: "shock",
    4: "others",
}

class POCDataPre:
    def __init__(self, sr=32000, target_len=30.75, repeat=None, empty_prob=0.):
        print(f'----- pre process poc data "{POC_DATA_PATH}" start. -----')
        print(f'empty_prob: {empty_prob}, repeat: {repeat}')

        self.root = POC_DATA_PATH
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
        self.process_normal_data()

        # abnormal
        self.process_abnormal_data(empty_prob)

        random.shuffle(self.train_data)
        random.shuffle(self.test_data)
        print(f'----- finish pre process data. -----')

    @property
    def train_set(self):
        return self.train_data

    @property
    def test_set(self):
        return self.test_data

    def summary(self):
        print(f'data summary:')
        for k, v in POC_CLASS_MAP.items():
            train_num = 0
            test_num = 0
            for audio in self.train_data:
                if audio["target"] == v:
                    train_num += 1
            for audio in self.test_data:
                if audio["target"] == v:
                    test_num += 1
            print(f"class {k}, train len {train_num}, test len {test_num}")

    def process_abnormal_data(self, empty_prob=0.):
        train_root = os.path.join(self.root, 'train', 'abnormal')
        test_root = os.path.join(self.root, 'test', 'abnormal')

        train_data = []
        for abnormal in os.listdir(train_root):
            # complete audio, generate by cut
            complete_dir = os.path.join(train_root, abnormal, 'complete')
            for d in os.listdir(complete_dir):
                temp_data = self.generate_wav_std(os.path.join(complete_dir, d), abnormal)
                train_data += temp_data

            # incomplete audio, generate by enhance
            cut_dir = os.path.join(train_root, abnormal, 'cut')
            for d in os.listdir(cut_dir):
                person_dir = os.path.join(cut_dir, d)
                if os.path.exists(os.path.join(person_dir, 'ori')):
                    temp_data = self.generate_wav_std(os.path.join(person_dir, 'ori'), abnormal)
                    train_data += temp_data
                temp_data = self.generate_wav_by_enhance(person_dir, abnormal, empty_prob=empty_prob)
                train_data += temp_data

        test_data = []
        for abnormal in os.listdir(test_root):
            # complete audio, generate by cut
            complete_dir = os.path.join(test_root, abnormal, 'complete')
            for d in os.listdir(complete_dir):
                temp_data = self.generate_wav_std(os.path.join(complete_dir, d), abnormal)
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
                    "target": POC_CLASS_MAP[target],
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

    def process_normal_data(self):
        train_root = os.path.join(self.root, 'train', 'normal')
        test_root = os.path.join(self.root, 'test', 'normal')

        train_data = self.generate_wav_std(train_root, 'normal')
        test_data = self.generate_wav_std(test_root, 'normal')

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
                    "target": POC_CLASS_MAP[target],
                    "waveform": signal
                })
        return results

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


def get_feature(model, dataset):
    feature_set = {}
    for item in tqdm(dataset, desc='calculate voice_print'):
        audio_class = POC_CLASS_DICT[item['target']]
        model_input = torch.Tensor(item['waveform']).unsqueeze(0)
        model_input = model_input.to(CUDA_DEVICE)
        del item['waveform']
        del item['target']

        with torch.no_grad():
            m_output = model(model_input)
            feature = m_output['feature'].cpu().numpy()

        feature_set[audio_class] = feature if audio_class not in feature_set.keys() \
            else np.concatenate((feature_set[audio_class], feature))

    print('generate voice_print finish.')
    return feature_set


def benchmark_poc(model, target_file=None):
    model = model.to(CUDA_DEVICE)
    model.eval()

    poc_data = POCDataPre(repeat={'shock': 2}, empty_prob=0.2)
    train_features = get_feature(model, poc_data.train_set)

    feature_center = {}
    for class_name, fts in train_features.items():
        feature_center[class_name] = cal_feature_center(fts)
    print('calculate voice_print center finish.')
    template_mat = np.expand_dims(feature_center['normal'], 0)
    for i in range(1, 4):
        template_mat = np.concatenate((template_mat, np.expand_dims(feature_center[POC_CLASS_DICT[i]], 0)))
    template_norm = np.linalg.norm(template_mat, axis=1)

    bmk_result = {c: [] for c in POC_CLASS_MAP.keys()}
    acc = 0.
    for test_data in tqdm(poc_data.test_set, f'infer test set'):
        with torch.no_grad():
            m_output = model(torch.Tensor(test_data['waveform']).unsqueeze(0).to(CUDA_DEVICE))
            target = m_output['feature'].cpu().squeeze(0).numpy()
            target_norm = np.linalg.norm(target)

        comp_vec = np.empty((len(template_mat)), dtype=np.float32)
        for i in range(comp_vec.shape[0]):
            comp_vec[i] = cos_similarity(template_mat[i], target, template_norm[i], target_norm)

        result = np.argmax(comp_vec)
        similar = np.max(comp_vec)
        if result == test_data['target']:
            acc += 1.
            bmk_result[POC_CLASS_DICT[test_data['target']]].append(np.around(similar, decimals=4))
    acc = np.around(acc / len(poc_data.test_set), decimals=4)
    print(f'benchmark acc: {acc}')
    for k, v in bmk_result.items():
        print(f'{k}: {v}')

    if target_file is not None:
        with open(target_file, 'w') as f:
            f.write(f'benchmark acc: {acc}\n')
            for k, v in bmk_result.items():
                f.write(f'{k}: {v}\n')


def test_main():
    config = parse_config()
    model = AutoEncoder(config)
    benchmark_poc(model)


if __name__ == '__main__':
    test_main()
