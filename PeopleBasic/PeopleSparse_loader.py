import numpy as np
# import scipy
# import sys
import os
import torch
import time

from scipy import sparse
# from scipy.sparse import csc_matrix
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

graph_data_path = '/home/miaohao/miao/Multi/Taxi_basic/'
# process = psutil.Process(os.getpid())
t1 = time.time()
batch_size = 40
split = [20*24, 5*24, 5*24]


def sp2Tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# class Sparse_DataSet(Dataset):
#     def __init__(self, img_path, label_path, mode=None):
#         # self.img_features = np.array(sparse.load_npz(img_path).todense())
#         self.img_features = sp2Tensor(sparse.load_npz(graph_data_path+img_path)).to_dense()
#         # self.label = np.array(sparse.load_npz(label_path).todense())
#         self.label = sp2Tensor(sparse.load_npz(graph_data_path+label_path)).to_dense()
#         # max_value = torch.max(torch.max(self.img_features), torch.max(self.label))
#         # print(max_value)
#         # print(self.img_features.shape, self.label.shape)
#         # print(sys.getsizeof(sparse.load_npz(img_path).todense())/(1024*1024*1024), sys.getsizeof(sparse.load_npz(
#         #     label_path).todense())/(1024**3))
#         # print(sys.getsizeof(self.img_features)/(1024*1024*1024), sys.getsizeof(self.label)/(1024**3))
#         if mode == 'od':
#             self.img_features = self.img_features.reshape((self.img_features.shape[0], 6, 16, 16, 256))
#             self.label = self.label.reshape((self.label.shape[0], 1, 16, 16, 256))
#         elif mode == 'node':
#             self.img_features = self.img_features.reshape((self.img_features.shape[0], 6, 273, 256))
#             self.label = self.label.reshape((self.label.shape[0], 1, 273, 256))
#         elif mode == 'adj':
#             self.img_features = self.img_features.reshape((self.img_features.shape[0], 6, 273, 273))
#             self.label = self.label.reshape((self.label.shape[0], 1, 273, 273))
#         # print(self.img_features.shape, self.label.shape)
#         # print(sys.getsizeof(self.img_features), sys.getsizeof(self.label))
#         # print()
#         # max_value = torch.max(torch.max(self.img_features), torch.max(self.label))
#         # print(max_value)

#     def __len__(self):
#         return len(self.img_features)

#     def __getitem__(self, item):
#         self.sample = self.img_features[item]
#         self.sample_label = self.label[item]
#         # print(sys.getsizeof(self.sample), sys.getsizeof(self.sample_label))
#         return self.sample, self.sample_label


class OD_sparse_dataset(Dataset):
    def __init__(self, img_path, label_path, mode=None):
        self.img_features = sp2Tensor(sparse.load_npz(img_path)).to_dense().reshape(-1, 3, 16, 16, 256)
        self.label = sp2Tensor(sparse.load_npz(label_path)).to_dense().reshape(-1, 1, 16, 16, 256)
        self.split = split
        self.mode = mode

        if self.mode == 'train':
            self.x_data = self.img_features[0:self.split[0]]
            self.y_data = self.label[0:self.split[0]]
            print(self.x_data.shape, self.y_data.shape)
        elif self.mode == 'validate':
            self.x_data = self.img_features[self.split[0]:self.split[0]+self.split[1]]
            self.y_data = self.label[self.split[0]:self.split[0]+self.split[1]]
        elif self.mode == 'test':
            self.x_data = self.img_features[self.split[0]+self.split[1]:]
            self.y_data = self.label[self.split[0]+self.split[1]:]
    
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, item):
        self.sample = self.x_data[item]
        self.sample_label = self.y_data[item]
        return self.sample, self.sample_label


class node_sparse_dataset(Dataset):
    def __init__(self, img_path, label_path, mode=None):
        self.img_features = sp2Tensor(sparse.load_npz(img_path)).to_dense().reshape(-1, 3, 273, 256)
        self.label = sp2Tensor(sparse.load_npz(label_path)).to_dense().reshape(-1, 1, 273, 256)
        self.split = split
        self.mode = mode

        if self.mode == 'train':
            self.x_data = self.img_features[0:self.split[0]]
            self.y_data = self.label[0:self.split[0]]
            print(self.x_data.shape, self.y_data.shape)
        elif self.mode == 'validate':
            self.x_data = self.img_features[self.split[0]:self.split[0]+self.split[1]]
            self.y_data = self.label[self.split[0]:self.split[0]+self.split[1]]
        elif self.mode == 'test':
            self.x_data = self.img_features[self.split[0]+self.split[1]:]
            self.y_data = self.label[self.split[0]+self.split[1]:]
    
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, item):
        self.sample = self.x_data[item]
        self.sample_label = self.y_data[item]
        return self.sample, self.sample_label


class adj_sparse_dataset(Dataset):
    def __init__(self, img_path, label_path, mode=None):
        self.img_features = sp2Tensor(sparse.load_npz(img_path)).to_dense().reshape(-1, 3, 273, 273)
        self.label = sp2Tensor(sparse.load_npz(label_path)).to_dense().reshape(-1, 1, 273, 273)
        self.split = split
        self.mode = mode

        if self.mode == 'train':
            self.x_data = self.img_features[0:self.split[0]]
            self.y_data = self.label[0:self.split[0]]
            print(self.x_data.shape, self.y_data.shape)
        elif self.mode == 'validate':
            self.x_data = self.img_features[self.split[0]:self.split[0]+self.split[1]]
            self.y_data = self.label[self.split[0]:self.split[0]+self.split[1]]
        elif self.mode == 'test':
            self.x_data = self.img_features[self.split[0]+self.split[1]:]
            self.y_data = self.label[self.split[0]+self.split[1]:]
    
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, item):
        self.sample = self.x_data[item]
        self.sample_label = self.y_data[item]
        return self.sample, self.sample_label



#
od_train_dataset = OD_sparse_dataset(img_path='/ceph_10826/halemiao/ft_local/od_img.npz', label_path='/ceph_10826/halemiao/ft_local/od_label.npz', mode='train')
od_train_dataloader = DataLoader(dataset=od_train_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
print('Over')
#
od_val_dataset = OD_sparse_dataset(img_path='/ceph_10826/halemiao/ft_local/od_img.npz', label_path='/ceph_10826/halemiao/ft_local/od_label.npz', mode='validate')
od_val_dataloader = DataLoader(dataset=od_val_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
print('Over')

od_test_dataset = OD_sparse_dataset(img_path='/ceph_10826/halemiao/ft_local/od_img.npz', label_path='/ceph_10826/halemiao/ft_local/od_label.npz', mode='test')
od_test_dataloader = DataLoader(dataset=od_test_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
print('Over')

node_train_dataset = node_sparse_dataset(img_path='/ceph_10826/halemiao/ft_local/node_features_data.npz', label_path='/ceph_10826/halemiao/ft_local/node_features_label.npz',
                                    mode='train')
node_train_dataloader = DataLoader(dataset=node_train_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
print('Over')

node_val_dataset = node_sparse_dataset(img_path='/ceph_10826/halemiao/ft_local/node_features_data.npz', label_path='/ceph_10826/halemiao/ft_local/node_features_label.npz', mode='validate')
node_val_dataloader = DataLoader(dataset=node_val_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
print('Over')

node_test_dataset = node_sparse_dataset(img_path='/ceph_10826/halemiao/ft_local/node_features_data.npz', label_path='/ceph_10826/halemiao/ft_local/node_features_label.npz', mode='test')
node_test_dataloader = DataLoader(dataset=node_test_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
print('Over')

adjs_train_dataset = adj_sparse_dataset(img_path='/ceph_10826/halemiao/ft_local/adjs_data.npz', label_path='/ceph_10826/halemiao/ft_local/adjs_label.npz', mode='train')
adjs_train_dataloader = DataLoader(dataset=adjs_train_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
print('Over')

adjs_val_dataset = adj_sparse_dataset(img_path='/ceph_10826/halemiao/ft_local/adjs_data.npz', label_path='/ceph_10826/halemiao/ft_local/adjs_label.npz', mode='validate')
adjs_val_dataloader = DataLoader(dataset=adjs_val_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
print('Over')

adjs_test_dataset = adj_sparse_dataset(img_path='/ceph_10826/halemiao/ft_local/adjs_data.npz', label_path='/ceph_10826/halemiao/ft_local/adjs_label.npz', mode='test')
adjs_test_dataloader = DataLoader(dataset=adjs_test_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
print('Over')
# print(sys.getsizeof(od_dataloader))
# process = psutil.Process(os.getpid())
# print('Memory', process.memory_info().rss/1024/1024/1024, 'GB')
t2 = time.time()
print('Dataloader finished!!! cost: {:.4f}min'.format((t2-t1)/60))
# for a, b in od_dataloader:
#     c = 1+1
