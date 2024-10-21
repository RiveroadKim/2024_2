import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import random

class TUTAcousticSceneDataset(Dataset):
    def __init__(self, data_dir, mode='train', apply_spec_augment=True):
        self.mode = mode
        self.apply_spec_augment = apply_spec_augment
        
        if self.mode == 'train':
            self.x_data = np.load(f"{data_dir}/X_train.npy")            
            tmp = pd.read_csv(f'{data_dir}/y_train.csv')['scene_label'].values            
            labels = ['beach', 'bus', 'cafe/restaurant', 'car', 'city center', 'forest path',
                      'grocery store', 'home', 'library', 'metro station', 'office', 'park',
                      'residential area', 'train', 'tram']
            self.label_encoder = {label: i for i, label in enumerate(labels)}
            self.y_data = np.array([self.label_encoder[label] for label in tmp])
        elif self.mode == 'test':
            self.x_data = np.load(f"{data_dir}/X_test.npy")
        else:
            raise ValueError("모드는 'train' 또는 'test'여야 합니다.")

    def __len__(self):
        return len(self.x_data)

    def spec_augment(self, spec, time_mask_ratio=0.1, freq_mask_ratio=0.1, n_time_masks=1, n_freq_masks=1):
        # 시간 마스킹
        t = spec.shape[2]
        t_mask_param = int(t * time_mask_ratio)
        for _ in range(n_time_masks):
            t0 = random.randint(0, t - t_mask_param)
            spec[:, :, t0:t0+t_mask_param] = 0

        # 주파수 마스킹
        f = spec.shape[1]
        f_mask_param = int(f * freq_mask_ratio)
        for _ in range(n_freq_masks):
            f0 = random.randint(0, f - f_mask_param)
            spec[:, f0:f0+f_mask_param, :] = 0

        return spec

    def __getitem__(self, idx):
        x = torch.from_numpy(self.x_data[idx]).float()
        x = x.unsqueeze(0)
        
        if self.mode == 'train' and self.apply_spec_augment:
            x = self.spec_augment(x)
        
        if self.mode == 'train':
            y = torch.tensor(self.y_data[idx]).long()
            return x, y
        else:
            return x

def get_dataloader(data_dir, mode='train', batch_size=32, shuffle=True, num_workers=0):
    dataset = TUTAcousticSceneDataset(data_dir, mode)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

# 사용 예시
if __name__ == "__main__":
    data_dir = "data"
    train_dataloader = get_dataloader(data_dir, mode='train', batch_size=32, shuffle=True)
    test_dataloader = get_dataloader(data_dir, mode='test', batch_size=32, shuffle=False)
    
    for batch_x, batch_y in train_dataloader:
        print("배치 크기:", batch_x.shape)
        print("레이블 크기:", batch_y.shape)
        break
    
    for batch_x in test_dataloader:
        print("배치 크기:", batch_x.shape)        
        break
