import torch
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

data = np.load('/ceph_10826/halemiao/ft_local/PeopleFlow_1616.npy')
data = data.reshape((-1, 32, 32, 2))[-30*24:]
# print(data[0, 12, 6, 0], data[0, 12, 6, 1])
print(data.max(), data.min())
data = data.reshape((-1, 16, 16, 2))
#data = (data - data.min())/(data.max() - data.min())
#print('Data Load over!', data.shape, data.max(), data.min())

data_split = [20*24, 5*24, 5*24]


def make_dataset(data_duration=3, label_duration=1, raw=data):
    x_data_list = []
    y_data_list = []
    for i in range(data_duration, len(raw)):
        x_data = raw[i-data_duration:i]
        # print(x_data.shape)
        x_data = x_data.reshape(1, -1, raw.shape[1], raw.shape[2], 2)
        # print(x_data.shape)
        x_data_list.append(x_data)

        y_data = raw[i:i+label_duration]
        y_data = y_data.reshape(1, -1, raw.shape[1], raw.shape[2], 2)
        # print(y_data.shape)
        y_data_list.append(y_data)
    x_data_list = np.concatenate(x_data_list, axis=0)
    y_data_list = np.concatenate(y_data_list, axis=0)
    return x_data_list, y_data_list


class PeopleDataset(Dataset):
    def __init__(self, mode=None, split=None):
        people_data_x, people_data_y = make_dataset(raw=data)
        people_data_x = people_data_x.transpose(0, 1, 4, 2, 3)
        # people_data_x = people_data_x.transpose(0, 4, 1, 2, 3)
        people_data_y = people_data_y.transpose(0, 1, 4, 2, 3)
        # people_data_y = people_data_y.transpose(0, 4, 1, 2, 3)
        self.mode = mode
        self.split = split

        if self.split is None:
            self.split = [20*24, 5*24, 5*24]
        
        if self.mode == 'train':
            self.x_data = people_data_x[0:self.split[0]]
            self.y_data = people_data_y[0:self.split[0]]
            print(self.x_data.shape, self.y_data.shape)
        elif self.mode == 'validate':
            self.x_data = people_data_x[self.split[0]:self.split[0]+self.split[1]]
            self.y_data = people_data_y[self.split[0]:self.split[0]+self.split[1]]
        elif self.mode == 'test':
            self.x_data = people_data_x[self.split[0]+self.split[1]:]
            self.y_data = people_data_y[self.split[0]+self.split[1]:]
    
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, item):
        self.sample_x = self.x_data[item]
        self.sample_y = self.y_data[item]
        return self.sample_x, self.sample_y


Flow_train_dataset = PeopleDataset(mode='train', split=data_split)
Flow_train_dataloader = DataLoader(dataset=Flow_train_dataset, batch_size=40, shuffle=True)

Flow_validate_dataset = PeopleDataset(mode='validate', split=data_split)
Flow_validate_dataloader = DataLoader(dataset=Flow_validate_dataset, batch_size=40, shuffle=False)

Flow_test_dataset = PeopleDataset(mode='test', split=data_split)
Flow_test_dataloader = DataLoader(dataset=Flow_test_dataset, batch_size=40, shuffle=False)