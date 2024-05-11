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
from flask import Flask,render_template
import numpy as np
import pandas as pd
import os
import pymysql
import datetime
import  json
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import deque
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from werkzeug.utils import secure_filename
from flask import Flask,render_template,request,redirect,url_for
from openpyxl import Workbook
import shutil, os
import datetime, time
from sklearn.metrics import r2_score,explained_variance_score,mean_absolute_error,f1_score


app = Flask(__name__)



@app.route('/')
def Login():
    def get_model(path: Union[str, Path]):
        path = Path(path)

        model_file = path if path.is_file() else get_best_model(path)
        if gpu:
            model_checkpoint = torch.load(model_file)
        else:
            model_checkpoint = torch.load(model_file, map_location='cpu')

        model = SUMO(config)
        model.load_state_dict(model_checkpoint['state_dict'])

        return model

    def get_best_model(experiment_path: Path, sort_by_loss: bool = False):
        models_path = experiment_path / 'models'
        models = list(models_path.glob('epoch=*.ckpt'))

        regex = r'.*val_loss=(0\.[0-9]+).*\.ckpt' if sort_by_loss else r'.*val_f1_mean=(0\.[0-9]+).*\.ckpt'
        regex_results = [re.search(regex, str(m)) for m in models]

        models_score = np.array([float(r.group(1)) for r in regex_results])
        model_idx = np.argmin(models_score) if sort_by_loss else np.argmax(models_score)

        return models[model_idx]

    class SimpleDataset(Dataset):
        def __init__(self, data_vectors):
            super(SimpleDataset, self).__init__()

            self.data = data_vectors

        def __len__(self) -> int:
            return len(self.data)

        @staticmethod
        def preprocess(data):
            return zscore(data)

        def __getitem__(self, idx):
            data = self.preprocess(self.data[idx])
            return torch.from_numpy(data).float(), torch.zeros(0)

    def A7(x, sr, return_features=False):
        thresholds = np.array([1.25, 1.3, 1.3, 0.69])
        win_length_sec = 0.3
        win_step_sec = 0.1
        features, spindles = detect_spindles(x, thresholds, win_length_sec, win_step_sec, sr)
        return spindles / sr if not return_features else (spindles / sr, features)

    def get_args():
        # synthetic input data
        default_data_path = (Path(__file__).absolute().parents[1] / 'input' / 'c3a1only.npy').__str__()
        default_model_path = (Path(__file__).absolute().parents[1] / 'output' / 'final.ckpt').__str__()

        parser = argparse.ArgumentParser(description='Evaluate a UTime model on any given eeg data')
        parser.add_argument('-d', '--data_path', type=str, default=default_data_path, help='Path to input data, given in \
        .pickle or .npy format as a dict with the channel name as key and the eeg data as value')
        parser.add_argument('-sr', '--sample_rate', type=float, default=100.0,
                            help='Rate with which the given data was sampled')
        parser.add_argument('-m', '--model_path', type=str, default=default_model_path,
                            help='Path to the model checkpoint used for evaluating')
        parser.add_argument('-g', '--gpu', action='store_true', default=False,
                            help='If a GPU should be used')

        return parser.parse_args()

    args = get_args()

    data_path = args.data_path
    sr = args.sample_rate
    model_path = args.model_path
    gpu = args.gpu

    resample_rate = 100.0

    config = Config('predict', create_dirs=False)

    eegs = [np.array(np.load(data_path, allow_pickle=True))]

    eegs = [downsample(butter_bandpass_filter(x, 0.3, 30.0, sr, 10), sr, resample_rate) for x in eegs]

    dataset = SimpleDataset(eegs)
    dataloader = DataLoader(dataset)

    model = get_model(model_path)

    trainer = pl.Trainer(gpus=int(gpu), num_sanity_val_steps=0, logger=False)
    predictions = trainer.predict(model, dataloader)
    fig, ax = plt.subplots(1, sharex=True)
    # area={}
    testuse = np.load("testuse.npy")
    # print(testuse)
    t = np.arange(eegs[0].shape[0]) / resample_rate
    area = {}
    # ax.plot(t, eegs[0], 'k-')


    spindle_vect = predictions[0][0].numpy()
    spindles = spindle_vect_to_indices(spindle_vect) / resample_rate
    area.setdefault("area", [list(x) for x in spindles])
    
    # area.setdefault(channel, [list(x) for x in spindles])/
    # print(spindle_vect)
    # for i, spindle in enumerate(spindles):
    #     ax.fill_between(spindle, -50, 50, alpha=0.3, color='orange')

    # for i, t in enumerate(testuse):
    #     ax.fill_between(t, -50, 50, alpha=0.3, color='blue', label="correct")

    #file = open("area.json", "w")
    #json.dump(area, file)
    #file.close()
    # fig.show()
    # plt.show()
    eeg=list(eegs[0])
    time=list(t)

    # print(eeg)
    # print(t)
    eeg_time=[]
    for i in range(0,len(eeg)):
        temp=[]
        temp.append(time[i])
        temp.append(eeg[i])
        eeg_time.append(temp)
    # print(area)
    space=area['area']
    for i in space:
        if(int(i[0]) == i[0]):
            i[0] = int(i[0])
    print(space)

    result=[]
    temp1=[]
    temp2={}
    for i in range(0,len(space)):
        for j in range(0,2):
            temp2['xAxis']=str(space[i][j])
            temp1.append(temp2)
            temp2={}
        result.append(temp1)
        temp1=[]
    
    return render_template("index.html",eeg=eeg,time=time,area=result)  #eeg是电压  time是时间  area是纺锤波范围

if __name__ == '__main__':
    app.run()
