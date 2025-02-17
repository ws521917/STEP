import datetime
import os

import gensim
import numpy as np
import pandas as pd
import torch
from gensim import models
from torch.utils.data import Dataset
from tqdm import tqdm


class MyDataset(Dataset):
    def __init__(self, config, dataset_path, device, load_mode):
        self.config = config
        self.device = device
        self.load_mode = load_mode
        self.dataset_path = dataset_path
        self.dataset='TC'in dataset_path
        self.user2id = np.load(os.path.join(dataset_path, 'user_mapper.npy'), allow_pickle=True).item()
        self.location2id = np.load(os.path.join(dataset_path, 'location_mapper.npy'), allow_pickle=True).item()

        if load_mode == 'test':
            self.data = self.load_npy_file(os.path.join(dataset_path, f'{load_mode}.npy'))
        else:
            if not os.path.exists(os.path.join(dataset_path, f'{load_mode}.npy')):
                self.generate_data(load_mode='train')
                self.generate_data(load_mode='test')
            self.data = self.load_npy_file(os.path.join(dataset_path, f'{load_mode}.npy'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]

        return data


    def generate_data(self, load_mode):
        occur_time_individual = np.load(os.path.join(self.dataset_path, f'occur_time_individual.npy'),
                                        allow_pickle=True)
        res = []
        with open(os.path.join(self.dataset_path, f'{load_mode}.csv'), 'r', encoding='utf8') as file:
            lines = file.readlines()
            for line_i, line in enumerate(tqdm(lines, desc=f'Initial {load_mode} data')):
                user = line.strip().split(',')[0]
                occur_time_user = occur_time_individual[self.user2id[user]]
                stay_points = line.strip().split(',')[1:]
 
                sequence_count, left = divmod(len(stay_points), self.config.Dataset.sequence_length)
                if sequence_count==0:
                    continue
                assert sequence_count > 0, f"{user}'s does not have enough data."
                sequence_count -= 1 if left == 0 else 0
                for i in range(sequence_count):
                    split_start = i * self.config.Dataset.sequence_length
                    split_end = (i + 1) * self.config.Dataset.sequence_length
                    location_x = [self.location2id[item.split('@')[0]] for item in stay_points[split_start:split_end]]
                    timestamp_x = [item.split('@')[1] for item in stay_points[split_start:split_end]]
                    location_y = [self.location2id[item.split('@')[0]] for item in
                                  stay_points[split_start + 1:split_end + 1]]
                    timestamp_y = [item.split('@')[1] for item in stay_points[split_start + 1:split_end + 1]]
                    if i==0:
                      timestamp_pre = [timestamp_x[0]] + [item.split('@')[1] for item in stay_points[split_start:split_end - 1]]
                    else:
                      timestamp_pre = [item.split('@')[1] for item in stay_points[split_start-1:split_end -1]]
                    timeslot_y = []
                    timeslot_x = []
                    weekday_x = []
                    weekend_y = []
                    for item1, item2 in zip(timestamp_x, timestamp_pre):
                        loc_dur = int(item1) - int(item2)
                        loc_dur = loc_dur # 确保时间戳是数值类型
                    for item in timestamp_x:
                        weekday, hour = datetime_to_features(item)
                        time_x = 24 * weekday + hour
                        if self.dataset:
                            timeslot_x.append(hour)
                        else:
                            timeslot_x.append(time_x)
                        weekday_x.append(weekday)
                    for item in timestamp_y:
                        weekday, hour = datetime_to_features(item)
                        time_y= weekday*24 +hour
                        if self.dataset:
                            timeslot_y.append(hour)
                        else:
                            timeslot_y.append(time_y)
                        weekend_y.append(weekday)
                    res.append(
                        {
                            'user': self.user2id[user],
                            'location_x': location_x,
                            'timeslot': timeslot_x,
                            'weekday': weekday_x,
                            'location_y': location_y,
                            'timeslot_y': timeslot_y,
                            'weekend_y': weekend_y,
                        }
                    )
        print(res[0])

        np.save(os.path.join(self.dataset_path, f'{load_mode}.npy'), res)

    def load_npy_file(self, save_path):
        loaded_data = np.load(save_path, allow_pickle=True)
        prob_matrix_time_individual = np.load(
            os.path.join(self.dataset_path, f'prob_matrix_time_individual.npy'),
            allow_pickle=True)
        for data in loaded_data:
            user_idx = data['user']
            data['prob_matrix_time_individual'] = prob_matrix_time_individual[user_idx]
        return loaded_data

def datetime_to_features(timestamp):
    dt = datetime.datetime.fromtimestamp(int(timestamp) // 1000)
    weekday = dt.weekday()
    hour = dt.hour
    return weekday, hour

