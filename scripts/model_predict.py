import argparse
import re
from pathlib import Path
from sys import path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from scipy.stats import zscore
from torch.utils.data import Dataset, DataLoader

from a7.butter_filter import butter_bandpass_filter, downsample
from a7.detect_spindles import detect_spindles

# append root dir to python path so that we find `sumo`
path.insert(0, str(Path(__file__).absolute().parents[1]))
from sumo.config import Config
from sumo.data import spindle_vect_to_indices
from sumo.model import SUMO
import numpy as np
import json


class model_predictor:

    def get_model(self,path: Union[str, Path]):
        path = Path(path)

        model_file = path if path.is_file() else self.get_best_model(path)
        if self.gpu:
            model_checkpoint = torch.load(model_file)
        else:
            model_checkpoint = torch.load(model_file, map_location='cpu')

        model = SUMO(self.config)
        model.load_state_dict(model_checkpoint['state_dict'])

        return model


    def get_best_model(self,experiment_path: Path, sort_by_loss: bool = False):
        models_path = experiment_path / 'models'
        models = list(models_path.glob('epoch=*.ckpt'))

        regex = r'.*val_loss=(0\.[0-9]+).*\.ckpt' if sort_by_loss else r'.*val_f1_mean=(0\.[0-9]+).*\.ckpt'
        regex_results = [re.search(regex, str(m)) for m in models]

        models_score = np.array([float(r.group(1)) for r in regex_results])
        model_idx = np.argmin(models_score) if sort_by_loss else np.argmax(models_score)

        return models[model_idx]


    class SimpleDataset(Dataset):
        def __init__(self, data_vectors):
            super().__init__()

            self.data = data_vectors

        def __len__(self) -> int:
            return len(self.data)

        @staticmethod
        def preprocess(data):
            return zscore(data)

        def __getitem__(self, idx):
            data = self.preprocess(self.data[idx])
            return torch.from_numpy(data).float(), torch.zeros(0)


    def A7(self,x, sr, return_features=False):
        thresholds = np.array([1.25, 1.3, 1.3, 0.69])
        win_length_sec = 0.3
        win_step_sec = 0.1
        features, spindles = detect_spindles(x, thresholds, win_length_sec, win_step_sec, sr)
        return spindles / sr if not return_features else (spindles / sr, features)

    def start(self,data_name,sample_rate):
        data_path = (Path(__file__).absolute().parents[1] / 'input' / data_name).__str__()
        resample_rate = sample_rate
        eegs = [np.array(np.load(data_path, allow_pickle=True))]
        eegs = [downsample(butter_bandpass_filter(x, 0.3, 30.0, sample_rate, 10), sample_rate, resample_rate) for x in eegs]
        dataset = self.SimpleDataset(eegs)
        dataloader = DataLoader(dataset)
        model = self.get_model(self.model_path)

        trainer = pl.Trainer(gpus=int(self.gpu), num_sanity_val_steps=0, logger=False)
        predictions = trainer.predict(model, dataloader)
        
        t = np.arange(eegs[0].shape[0]) / resample_rate
        area = {}

        spindle_vect = predictions[0][0].numpy()
        spindles = spindle_vect_to_indices(spindle_vect) / resample_rate
        area.setdefault("area",[list(x) for x in spindles])
        file = open("newdata_model.json","w")
        json.dump(area,file)
        file.close()

    def __init__(self):
        self.model_path = (Path(__file__).absolute().parents[1] / 'output' / 'final.ckpt').__str__()
        self.gpu = False
        self.config = Config('predict', create_dirs=False)