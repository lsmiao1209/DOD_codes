import time

import math
import os

import matplotlib.pyplot as plt
import torch
import pandas as pd
import sklearn.cluster
seed = 42
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
import random
import numpy as np
import scipy
import sklearn
import sklearn.decomposition
import sklearn.manifold
import sklearn.metrics
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from Tools.hypergraph import Hypergraph
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture

from copulas.multivariate import VineCopula
from copulas.univariate import GaussianKDE


dataname_list = ['WPBC', 'wdbc','breastw', 'pima', 'cardio', 'cardiotocography', 'thyroid',
'Stamps', 'SpamBase', 'kddcup99','http','wine', 'musk','fault','waveform', 'CIFAR10_0', 'mnist','celeba', 'glass','yeast',
'speech','wilt', 'landsat','imdb', 'campaign','census']



def seed_torch(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


def getdataNN(dataname, rato):

    data = np.load(f'./datasets/{dataname}.npz', allow_pickle=True)

    label = data['y'].astype('float32')
    data = data['X'].astype('float32')


    # if the dataset is too large, subsampling for considering the computational cost
    if len(label) >= 10000:
        idx_sample = np.random.choice(np.arange(len(label)), 10000, replace=False)
        data = data[idx_sample]
        label = label[idx_sample]

    normal_data = data[label == 0]
    normal_label = label[label == 0]
    anom_data = data[label == 1]
    anom_label = label[label == 1]

    test_idy = np.random.choice(np.arange(0, len(anom_data)), int(len(anom_data)*rato), replace=False)
    test_idx = np.random.choice(np.arange(0, len(normal_data)), int(len(normal_data)*rato), replace=False)

    train_idx = np.setdiff1d(np.arange(0, len(normal_data)), test_idx)

    train_x = normal_data[train_idx]
    train_y = normal_label[train_idx]

    test_x = np.concatenate((normal_data[test_idx], anom_data[test_idy]))
    test_y = np.concatenate((normal_label[test_idx], anom_label[test_idy]))

    train_x, train_y = shuffle(train_x, train_y)
    test_x, test_y = shuffle(test_x, test_y)

    return train_x, train_y, test_x, test_y



def getdataNN_param(dataname, rato):

    data = np.load(f'./datasets/{dataname}.npz', allow_pickle=True)

    label = data['y'].astype('float32')
    data = data['X'].astype('float32')

    # Standardization
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # if the dataset is too large, subsampling for considering the computational cost
    if len(label) >= 10000:
        idx_sample = np.random.choice(np.arange(len(label)), 10000, replace=False)
        data = data[idx_sample]
        label = label[idx_sample]

    normal_data = data[label == 0]
    normal_label = label[label == 0]
    anom_data = data[label == 1]
    anom_label = label[label == 1]

    test_idy = np.random.choice(np.arange(0, len(anom_data)), int(len(anom_data)*rato), replace=False)
    test_idx = np.random.choice(np.arange(0, len(normal_data)), int(len(normal_data)*rato), replace=False)

    train_idx = np.setdiff1d(np.arange(0, len(normal_data)), test_idx)

    train_x = normal_data[train_idx]
    train_y = normal_label[train_idx]

    test_x = np.concatenate((normal_data[test_idx], anom_data[test_idy]))
    test_y = np.concatenate((normal_label[test_idx], anom_label[test_idy]))

    train_x, train_y = shuffle(train_x, train_y)
    # tips 训练 waveform注释掉shuttle 效果可以
    test_x, test_y = shuffle(test_x, test_y)



    return train_x, train_y, test_x, test_y



def getdataNN_cofd(dataname, rato):

    data = np.load(f'./datasets/{dataname}.npz', allow_pickle=True)
    label = data['y'].astype('float32')
    data = data['X'].astype('float32')
    # Standardization
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # if the dataset is too large, subsampling for considering the computational cost
    if len(label) >= 10000:
        idx_sample = np.random.choice(np.arange(len(label)), 10000, replace=False)
        data = data[idx_sample]
        label = label[idx_sample]

    normal_data = data[label == 0]
    normal_label = label[label == 0]
    anom_data = data[label == 1]
    anom_label = label[label == 1]

    test_idy = np.random.choice(np.arange(0, len(anom_data)), int(len(anom_data)*rato), replace=False)
    test_idx = np.random.choice(np.arange(0, len(normal_data)), int(len(normal_data)*rato), replace=False)

    train_idx = np.setdiff1d(np.arange(0, len(normal_data)), test_idx)
    train_idy = np.setdiff1d(np.arange(0, len(anom_data)), test_idy)

    train_x = normal_data[train_idx]
    train_y = normal_label[train_idx]

    # train_x = np.concatenate((normal_data[train_idx], anom_data[train_idy]))
    # train_y = np.concatenate((normal_label[train_idx], anom_label[train_idy]))

    test_x = np.concatenate((normal_data[test_idx], anom_data[test_idy]))
    test_y = np.concatenate((normal_label[test_idx], anom_label[test_idy]))

    train_x, train_y = shuffle(train_x, train_y)
    test_x, test_y = shuffle(test_x, test_y)

    train_x_neg = anom_data[train_idy]
    train_y_neg  = anom_label[train_idy]

    return train_x, train_y, test_x, test_y,train_x_neg,train_y_neg





def getdataNN_all(dataname, ratio):
    data = np.load(f'./datasets/{dataname}.npz', allow_pickle=True)
    label = data['y'].astype('float32')
    data = data['X'].astype('float32')

    # Standardization
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # If the dataset is too large, subsampling for considering the computational cost
    if len(label) >= 10000:
        idx_sample = np.random.choice(np.arange(len(label)), 10000, replace=False)
        data = data[idx_sample]
        label = label[idx_sample]

    normal_data = data[label == 0]
    normal_label = label[label == 0]
    anom_data = data[label == 1]
    anom_label = label[label == 1]

    # Split normal data for training based on ratio
    train_idx = np.random.choice(np.arange(0, len(normal_data)), int(len(normal_data) * ratio), replace=False)
    test_idx = np.setdiff1d(np.arange(0, len(normal_data)), train_idx)

    train_x = normal_data[train_idx]
    train_y = normal_label[train_idx]

    # Test data includes remaining normal data and all anomaly data
    test_x = np.concatenate((normal_data[test_idx], anom_data))
    test_y = np.concatenate((normal_label[test_idx], anom_label))

    train_x, train_y = shuffle(train_x, train_y)
    test_x, test_y = shuffle(test_x, test_y)

    return train_x, train_y, test_x, test_y










def getdataNN_SYN_save_data(dataname, types):

    data = np.load(f'./datasets/{dataname}.npz', allow_pickle=True)
    label = data['y'].astype('float32')
    data = data['X'].astype('float32')

    # Standardization
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # if the dataset is too large, subsampling for considering the computational cost
    if len(label) >= 1000:
        idx_sample = np.random.choice(np.arange(len(label)), 1000, replace=False)
        data = data[idx_sample]
        label = label[idx_sample]

    data, label = generate_realistic_synthetic(data, label, types, 5, 1.1)

    return data, label

def tsne(data, dim):
    pca = sklearn.decomposition.PCA(n_components=dim)
    pca_result = pca.fit_transform(data)

    tsne = sklearn.manifold.TSNE(n_components=dim, perplexity=30, n_iter=1000, learning_rate='auto', random_state=20150101,
                                 init=pca_result)
    X_tsne = tsne.fit_transform(data)

    # Standardization
    scaler = sklearn.preprocessing.StandardScaler()
    X_tsne = scaler.fit_transform(X_tsne)

    return X_tsne

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def train_test(data, label):

    normal_data = data[label == 0]
    normal_label = label[label == 0]
    anom_data = data[label == 1]
    anom_label = label[label == 1]

    test_idy = np.random.choice(np.arange(0, len(anom_data)), int(len(anom_data)*0.2), replace=False)
    test_idx = np.random.choice(np.arange(0, len(normal_data)), int(len(normal_data)*0.2), replace=False)

    train_idx = np.setdiff1d(np.arange(0, len(normal_data)), test_idx)

    train_x = normal_data[train_idx]
    train_y = normal_label[train_idx]

    test_x = np.concatenate((normal_data[test_idx], anom_data[test_idy]))
    test_y = np.concatenate((normal_label[test_idx], anom_label[test_idy]))

    train_x, train_y = shuffle(train_x, train_y)
    test_x, test_y = shuffle(test_x, test_y)

    return test_x, test_y


def save_data(data,label,dataname,data_type):
    # dataname_list = ['WPBC', 'wdbc', 'breastw', 'pima', 'cardio', 'Stamps',
    #                  'SpamBase', 'glass', 'yeast', 'wilt',
    #                  'landsat'
    #                  ]
    result_axis0 = np.concatenate((data, label.reshape(-1, 1)), axis=1)
    save_dir = f'./datasets/generated_data/{data_type}'
    np.save(os.path.join(save_dir, f'{dataname}.npy'), result_axis0)
    print(dataname)

def getdataNN_SYN_train_test(dataname, types,c_alg):

    datasss = np.load(f'./datasets/generated_data/{types}/{dataname}.npy')

    data = datasss[:, :-1]
    label = datasss[:, -1]


    normal_data = data[label == 0]
    normal_label = label[label == 0]
    anom_data = data[label == 1]
    anom_label = label[label == 1]

    test_idy = np.random.choice(np.arange(0, len(anom_data)), int(len(anom_data)*0.2), replace=False)
    test_idx = np.random.choice(np.arange(0, len(normal_data)), int(len(normal_data)*0.2), replace=False)

    train_idx = np.setdiff1d(np.arange(0, len(normal_data)), test_idx)

    train_x = normal_data[train_idx]
    train_y = normal_label[train_idx]

    test_x = np.concatenate((normal_data[test_idx], anom_data[test_idy]))
    test_y = np.concatenate((normal_label[test_idx], anom_label[test_idy]))

    train_x, train_y = shuffle(train_x, train_y)
    test_x, test_y = shuffle(test_x, test_y)

    return train_x, train_y, test_x, test_y


def getdataTime(size):

    data = np.load(f'../datasets/celeba.npz', allow_pickle=True)
    label = data['y'].astype('float32')
    data = data['X'].astype('float32')
    # Standardization
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    idx_sample = np.random.choice(np.arange(len(label)), size, replace=False)
    data = data[idx_sample]
    label = label[idx_sample]


    train_x = data
    train_y = label

    test_x = data
    test_y = label

    train_x, train_y = shuffle(train_x, train_y)
    test_x, test_y = shuffle(test_x, test_y)

    return train_x, train_y, test_x, test_y


def getdata_dimension(dataname, rato,m):

    data = np.load(f'../datasets/{dataname}.npz', allow_pickle=True)
    label = data['y'].astype('float32')
    data = data['X'].astype('float32')
    # Standardization
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    d = data.shape[1]

    selected_indices = np.random.choice(d, m, replace=False)  # 随机选择m个维度索引，确保不重复
    data = data[:, selected_indices]  # 提取对应的维度数据


    train_x = data
    train_y = label

    test_x = data
    test_y = label

    train_x, train_y = shuffle(train_x, train_y)
    test_x, test_y = shuffle(test_x, test_y)

    return train_x, train_y, test_x, test_y



def add_corruptted(X, y, missing_ratio:float):
    n, d = X.shape
    mask = np.random.rand(n, d)
    mask = (mask > missing_ratio).astype(float)
    if missing_ratio > 0.0:
        X[mask == 0] = np.nan
        imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        X = imputer.fit_transform(X)
    return X, y

def add_irrelevant_features(X, y, noise_ratio:float):
    # adding uniform noise
    noise_dim = int(noise_ratio * X.shape[1])
    if noise_ratio == 0.0:
        pass
    else:
        if noise_ratio == 0.1:
            noise_dim = 1
        if noise_dim > 0:
            X_noise = []
            for i in range(noise_dim):
                idx = np.random.choice(np.arange(X.shape[1]), 1)
                X_min = np.min(X[:, idx])
                X_max = np.max(X[:, idx])

                X_noise.append(np.random.uniform(X_min, X_max, size=(X.shape[0], 1)))

            # concat the irrelevant noise feature
            X_noise = np.hstack(X_noise)
            X = np.concatenate((X, X_noise), axis=1)
            # shuffle the dimension
            idx = np.random.choice(np.arange(X.shape[1]), X.shape[1], replace=False)
            X = X[:, idx]

    return X, y
def add_duplicated_anomalies(X, y, duplicate_times: int):
    if duplicate_times <= 1:
        pass
    else:
        # index of normal and anomaly data
        idx_n = np.where(y == 0)[0]
        idx_a = np.where(y == 1)[0]

        # generate duplicated anomalies
        idx_a = np.random.choice(idx_a, int(len(idx_a) * duplicate_times))

        idx = np.append(idx_n, idx_a);
        random.shuffle(idx)
        X = X[idx];
        y = y[idx]

    return X, y

def add_noisy_data(X, y, noise_ratio: int):
    if noise_ratio == 0.0:
        pass
    else:
        # noise = np.random.randn(X.shape[0], X.shape[1])
        # X = X + noise_ratio * noise

        # negative sample hyperparameters
        epsilon = 1
        proportion = 1
        # M
        randmat = np.random.rand(X.shape[0], X.shape[1]) < noise_ratio
        #  subspace perturbation samples
        X = np.tile(X, (proportion, 1)) + randmat * (epsilon * np.random.randn(X.shape[0], X.shape[1]))

    return X, y





def getDatas(dataname, rato, type):
    r"""
    Args:
        ``rato`` (``float``): Ratio of normal data to all normal data in the test set (0.3).
        ``type`` (``string``): type of noise in the data.
        ``Normal``: Indicates normal data, with 0.7 normal data used for training, 0.3 normal data and 0.5 abnormal data used for testing.

        ``Anomaly1`` : Indicates training data with 0.1 anomaly.
        ``Anomaly3``: indicates that there are 0.3 anomalies in the training data.
        ``Anomaly5``: indicates an anomaly of 0.5 in the training data.

        ``irrfea1``: indicates that there are 0.1*dim uncorrelated features in the training and test data.
        ``irrfea3``: Indicates that there are 0.3*dim uncorrelated features in the training and test data.
        ``irrfea5``: indicates that there are 0.5*dim uncorrelated features in the training and test data.

        ``corrupt1``: indicates that there is a missing value of 0.1 ratio in the training and test data.
        ``corrupt3``: Indicates that there is a 0.3 ratio of missing values in the training and test data.
        ``corrupt5``: Indicates that there are missing values in the training and test data with a ratio of 0.5.
    """

    data = np.load(f'../datasets/{dataname}.npz', allow_pickle=True)
    label = data['y'].astype('float32')
    data = data['X'].astype('float32')

    scaler = StandardScaler()
    data = scaler.fit_transform(data)


    normal_data = data[label == 0]
    normal_label = label[label == 0]
    anom_data = data[label == 1]
    anom_label = label[label == 1]



    test_idx = np.random.choice(np.arange(0, len(normal_data)), int(len(normal_data)*rato), replace=False)
    test_idy = np.random.choice(np.arange(0, len(anom_data)), int(len(anom_data) * rato), replace=False)

    train_idx = np.setdiff1d(np.arange(0, len(normal_data)), test_idx)
    # tips 用作添加异常值的数据
    train_idy = np.setdiff1d(np.arange(0, len(anom_data)), test_idy)

    abnor_id1 = np.random.choice(np.arange(0, len(train_idy)), max(1, int(len(train_idx)*0.001)), replace=False)
    abnor_id2 = np.random.choice(np.arange(0, len(train_idy)), max(1, int(len(train_idx) * 0.05)), replace=False)
    abnor_id3 = np.random.choice(np.arange(0, len(train_idy)), int(len(train_idx) * 0.1), replace=False)
    abnor_id4 = np.random.choice(np.arange(0, len(train_idy)), max(1, int(len(train_idx) * 0.3)), replace=False)
    abnor_id5 = np.random.choice(np.arange(0, len(train_idy)), max(1, int(len(train_idx) * 0.5)), replace=False)


    if type == 'Normal':
        train_x = normal_data[train_idx]
        train_y = normal_label[train_idx]

        test_x = np.concatenate((normal_data[test_idx], anom_data[test_idy]))
        test_y = np.concatenate((normal_label[test_idx], anom_label[test_idy]))
        #
        train_x, train_y = shuffle(train_x, train_y)
        test_x, test_y = shuffle(test_x, test_y)

    elif type == 'Anomaly1':
        train_x = np.concatenate((normal_data[train_idx], anom_data[abnor_id1]))
        train_y = np.concatenate((normal_label[train_idx], anom_label[abnor_id1]))
        train_y = np.zeros_like(train_y)

        test_x = np.concatenate((normal_data[test_idx], anom_data[test_idy]))
        test_y = np.concatenate((normal_label[test_idx], anom_label[test_idy]))

        train_x, train_y = shuffle(train_x, train_y)
        test_x, test_y = shuffle(test_x, test_y)

    elif type == 'Anomaly2':
        train_x = np.concatenate((normal_data[train_idx], anom_data[abnor_id2]))
        train_y = np.concatenate((normal_label[train_idx], anom_label[abnor_id2]))
        train_y = np.zeros_like(train_y)

        test_x = np.concatenate((normal_data[test_idx], anom_data[test_idy]))
        test_y = np.concatenate((normal_label[test_idx], anom_label[test_idy]))

        train_x, train_y = shuffle(train_x, train_y)
        test_x, test_y = shuffle(test_x, test_y)

    elif type == 'Anomaly3':
        train_x = np.concatenate((normal_data[train_idx], anom_data[abnor_id3]))
        train_y = np.concatenate((normal_label[train_idx], anom_label[abnor_id3]))
        train_y = np.zeros_like(train_y)

        test_x = np.concatenate((normal_data[test_idx], anom_data[test_idy]))
        test_y = np.concatenate((normal_label[test_idx], anom_label[test_idy]))

        train_x, train_y = shuffle(train_x, train_y)
        test_x, test_y = shuffle(test_x, test_y)

    elif type == 'Anomaly4':
        train_x = np.concatenate((normal_data[train_idx], anom_data[abnor_id4]))
        train_y = np.concatenate((normal_label[train_idx], anom_label[abnor_id4]))
        train_y = np.zeros_like(train_y)

        test_x = np.concatenate((normal_data[test_idx], anom_data[test_idy]))
        test_y = np.concatenate((normal_label[test_idx], anom_label[test_idy]))

        train_x, train_y = shuffle(train_x, train_y)
        test_x, test_y = shuffle(test_x, test_y)

    elif type == 'Anomaly5':
        train_x = np.concatenate((normal_data[train_idx], anom_data[abnor_id5]))
        train_y = np.concatenate((normal_label[train_idx], anom_label[abnor_id5]))
        train_y = np.zeros_like(train_y)

        test_x = np.concatenate((normal_data[test_idx], anom_data[test_idy]))
        test_y = np.concatenate((normal_label[test_idx], anom_label[test_idy]))

        train_x, train_y = shuffle(train_x, train_y)
        test_x, test_y = shuffle(test_x, test_y)



    elif type == 'irrfea1':
        train_x = normal_data[train_idx]
        train_y = normal_label[train_idx]

        test_x = np.concatenate((normal_data[test_idx], anom_data[test_idy]))
        test_y = np.concatenate((normal_label[test_idx], anom_label[test_idy]))

        train_x, train_y = shuffle(train_x, train_y)
        test_x, test_y = shuffle(test_x, test_y)

        train_x, train_y = add_irrelevant_features(train_x, train_y, 0.001)
        test_x, test_y = add_irrelevant_features(test_x, test_y, 0.001)

    elif type == 'irrfea2':
        train_x = normal_data[train_idx]
        train_y = normal_label[train_idx]

        test_x = np.concatenate((normal_data[test_idx], anom_data[test_idy]))
        test_y = np.concatenate((normal_label[test_idx], anom_label[test_idy]))

        train_x, train_y = shuffle(train_x, train_y)
        test_x, test_y = shuffle(test_x, test_y)

        train_x, train_y = add_irrelevant_features(train_x, train_y, 0.05)
        test_x, test_y = add_irrelevant_features(test_x, test_y, 0.05)

    elif type == 'irrfea3':
        train_x = normal_data[train_idx]
        train_y = normal_label[train_idx]

        test_x = np.concatenate((normal_data[test_idx], anom_data[test_idy]))
        test_y = np.concatenate((normal_label[test_idx], anom_label[test_idy]))

        train_x, train_y = shuffle(train_x, train_y)
        test_x, test_y = shuffle(test_x, test_y)

        train_x, train_y = add_irrelevant_features(train_x, train_y, 0.1)
        test_x, test_y = add_irrelevant_features(test_x, test_y, 0.1)

    elif type == 'irrfea4':
        train_x = normal_data[train_idx]
        train_y = normal_label[train_idx]

        test_x = np.concatenate((normal_data[test_idx], anom_data[test_idy]))
        test_y = np.concatenate((normal_label[test_idx], anom_label[test_idy]))

        train_x, train_y = shuffle(train_x, train_y)
        test_x, test_y = shuffle(test_x, test_y)

        train_x, train_y = add_irrelevant_features(train_x, train_y, 0.3)
        test_x, test_y = add_irrelevant_features(test_x, test_y, 0.3)

    elif type == 'irrfea5':
        train_x = normal_data[train_idx]
        train_y = normal_label[train_idx]

        test_x = np.concatenate((normal_data[test_idx], anom_data[test_idy]))
        test_y = np.concatenate((normal_label[test_idx], anom_label[test_idy]))

        train_x, train_y = shuffle(train_x, train_y)
        test_x, test_y = shuffle(test_x, test_y)

        train_x, train_y = add_irrelevant_features(train_x, train_y, 0.5)
        test_x, test_y = add_irrelevant_features(test_x, test_y, 0.5)


    elif type == 'corrupt1':
        train_x = normal_data[train_idx]
        train_y = normal_label[train_idx]

        test_x = np.concatenate((normal_data[test_idx], anom_data[test_idy]))
        test_y = np.concatenate((normal_label[test_idx], anom_label[test_idy]))

        train_x, train_y = shuffle(train_x, train_y)
        test_x, test_y = shuffle(test_x, test_y)

        train_x, train_y = add_corruptted(train_x, train_y, 0.001)
        test_x, test_y = add_corruptted(test_x, test_y, 0.001)

    elif type == 'corrupt2':
        train_x = normal_data[train_idx]
        train_y = normal_label[train_idx]

        test_x = np.concatenate((normal_data[test_idx], anom_data[test_idy]))
        test_y = np.concatenate((normal_label[test_idx], anom_label[test_idy]))

        train_x, train_y = shuffle(train_x, train_y)
        test_x, test_y = shuffle(test_x, test_y)

        train_x, train_y = add_corruptted(train_x, train_y, 0.05)
        test_x, test_y = add_corruptted(test_x, test_y, 0.05)


    elif type == 'corrupt3':
        train_x = normal_data[train_idx]
        train_y = normal_label[train_idx]

        test_x = np.concatenate((normal_data[test_idx], anom_data[test_idy]))
        test_y = np.concatenate((normal_label[test_idx], anom_label[test_idy]))

        train_x, train_y = shuffle(train_x, train_y)
        test_x, test_y = shuffle(test_x, test_y)

        train_x, train_y = add_corruptted(train_x, train_y, 0.1)
        test_x, test_y = add_corruptted(test_x, test_y, 0.1)

    elif type == 'corrupt4':
        train_x = normal_data[train_idx]
        train_y = normal_label[train_idx]

        test_x = np.concatenate((normal_data[test_idx], anom_data[test_idy]))
        test_y = np.concatenate((normal_label[test_idx], anom_label[test_idy]))

        train_x, train_y = shuffle(train_x, train_y)
        test_x, test_y = shuffle(test_x, test_y)

        train_x, train_y = add_corruptted(train_x, train_y, 0.3)
        test_x, test_y = add_corruptted(test_x, test_y, 0.3)

    elif type == 'corrupt5':
        train_x = normal_data[train_idx]
        train_y = normal_label[train_idx]

        test_x = np.concatenate((normal_data[test_idx], anom_data[test_idy]))
        test_y = np.concatenate((normal_label[test_idx], anom_label[test_idy]))

        train_x, train_y = shuffle(train_x, train_y)
        test_x, test_y = shuffle(test_x, test_y)

        train_x, train_y = add_corruptted(train_x, train_y, 0.5)
        test_x, test_y = add_corruptted(test_x, test_y, 0.5)


    elif type == 'noise1':
        train_x = normal_data[train_idx]
        train_y = normal_label[train_idx]

        test_x = np.concatenate((normal_data[test_idx], anom_data[test_idy]))
        test_y = np.concatenate((normal_label[test_idx], anom_label[test_idy]))

        train_x, train_y = shuffle(train_x, train_y)
        test_x, test_y = shuffle(test_x, test_y)

        train_x, train_y = add_noisy_data(train_x, train_y, 0.001)
        test_x, test_y = add_noisy_data(test_x, test_y, 0.001)

    elif type == 'noise2':
        train_x = normal_data[train_idx]
        train_y = normal_label[train_idx]

        test_x = np.concatenate((normal_data[test_idx], anom_data[test_idy]))
        test_y = np.concatenate((normal_label[test_idx], anom_label[test_idy]))

        train_x, train_y = shuffle(train_x, train_y)
        test_x, test_y = shuffle(test_x, test_y)

        train_x, train_y = add_noisy_data(train_x, train_y, 0.05)
        test_x, test_y = add_noisy_data(test_x, test_y, 0.05)

    elif type == 'noise3':
        train_x = normal_data[train_idx]
        train_y = normal_label[train_idx]

        test_x = np.concatenate((normal_data[test_idx], anom_data[test_idy]))
        test_y = np.concatenate((normal_label[test_idx], anom_label[test_idy]))

        train_x, train_y = shuffle(train_x, train_y)
        test_x, test_y = shuffle(test_x, test_y)

        train_x, train_y = add_noisy_data(train_x, train_y, 0.1)
        test_x, test_y = add_noisy_data(test_x, test_y, 0.1)

    elif type == 'noise4':
        train_x = normal_data[train_idx]
        train_y = normal_label[train_idx]

        test_x = np.concatenate((normal_data[test_idx], anom_data[test_idy]))
        test_y = np.concatenate((normal_label[test_idx], anom_label[test_idy]))

        train_x, train_y = shuffle(train_x, train_y)
        test_x, test_y = shuffle(test_x, test_y)

        train_x, train_y = add_noisy_data(train_x, train_y, 0.3)
        test_x, test_y = add_noisy_data(test_x, test_y, 0.3)

    elif type == 'noise5':
        train_x = normal_data[train_idx]
        train_y = normal_label[train_idx]

        test_x = np.concatenate((normal_data[test_idx], anom_data[test_idy]))
        test_y = np.concatenate((normal_label[test_idx], anom_label[test_idy]))

        train_x, train_y = shuffle(train_x, train_y)
        test_x, test_y = shuffle(test_x, test_y)

        train_x, train_y = add_noisy_data(train_x, train_y, 0.5)
        test_x, test_y = add_noisy_data(test_x, test_y, 0.5)

    return train_x, train_y, test_x, test_y

def shuffle(X, Y):
    """
    Shuffle the datasets
    Args:
        X: input data
        Y: labels

    Returns: shuffled sets
    """
    random_index = np.random.permutation(X.shape[0])
    return X[random_index], Y[random_index]

def convert(args, device, k_nebor, train_x, train_y, test_x, test_y,w_list,mad):

    # tensor
    train_x = torch.tensor(train_x).to(device).float()
    train_y = torch.tensor(train_y).to(device).long()
    test_x = torch.tensor(test_x).to(device).float()
    test_y = torch.tensor(test_y).to(device).long()

    if w_list == 't' and mad == 't':
        s0 = time.time()
        hg_train = from_feature_kNN(args,train_x, k=k_nebor,w_list=True, mad=True)
        hg_train = hg_train.to(device)
        train_g_time = time.time() -s0

        s1 = time.time()
        hg_test = from_feature_kNN(args,test_x, k=k_nebor,w_list=True, mad=True)
        test_g_time = time.time() - s1
        hg_test = hg_test.to(device)


    elif w_list == 't'and mad == 'f':
        s0 = time.time()
        hg_train = from_feature_kNN(args,train_x, k=k_nebor,w_list=True, mad=False)
        hg_train = hg_train.to(device)
        train_g_time = time.time() - s0
        s1= time.time()
        hg_test = from_feature_kNN(args,test_x, k=k_nebor,w_list=True, mad=False)
        test_g_time = time.time() - s1
        hg_test = hg_test.to(device)


    elif w_list == 'f':
        s0 = time.time()
        hg_train = from_feature_kNN(args,train_x, k=k_nebor, w_list=False)
        hg_train = hg_train.to(device)
        train_g_time = time.time() - s0
        s1 = time.time()
        hg_test = from_feature_kNN(args,test_x, k=k_nebor, w_list=False)
        test_g_time = time.time() - s1
        hg_test = hg_test.to(device)


    return train_x, train_y, test_x, test_y, hg_train, hg_test,train_g_time,test_g_time


def Metrics(test_y, error):
    auc = roc_auc_score(test_y, error)
    pr = sklearn.metrics.average_precision_score(test_y, error)
    return auc, pr

def CalMetrics(test_x, test_y, error):
    auc = roc_auc_score(test_y.cpu(), error.cpu())
    pr = sklearn.metrics.average_precision_score(test_y.cpu(), error.cpu())

    return auc, pr

def printResults(dataname, auclist, prlist, tims):

    max_index = np.argmax(auclist)
    auc = auclist[max_index]
    pr = prlist[max_index]

    return auc, pr, tims

def _e_list_from_feature_kNN(features: torch.Tensor, k: int):
    r"""Construct hyperedges from the feature matrix. Each hyperedge in the hypergraph is constructed by the central vertex ans its :math:`k-1` neighbor vertices.

    Args:
        ``features`` (``torch.Tensor``): The feature matrix.
        ``k`` (``int``): The number of nearest neighbors.
    """
    features = features.cpu().numpy()

    assert features.ndim == 2, "The feature matrix should be 2-D."
    assert (
            k <= features.shape[0]
    ), "The number of nearest neighbors should be less than or equal to the number of vertices."

    # cKDTree
    tree = scipy.spatial.cKDTree(features)
    dist, nbr_array = tree.query(features, k=k)
    return dist, nbr_array.tolist()

def _e_list_from_feature_kNN11(features: torch.Tensor, k: int):
    r"""Construct hyperedges from the feature matrix. Each hyperedge in the hypergraph is constructed by the central vertex ans its :math:`k-1` neighbor vertices.

    Args:
        ``features`` (``torch.Tensor``): The feature matrix.
        ``k`` (``int``): The number of nearest neighbors.
    """
    # features = features.cpu().numpy()
    #
    # assert features.ndim == 2, "The feature matrix should be 2-D."
    # assert (
    #         k <= features.shape[0]
    # ), "The number of nearest neighbors should be less than or equal to the number of vertices."
    #
    # # cKDTree
    # tree = scipy.spatial.cKDTree(features)
    # dist, nbr_array = tree.query(features, k=k)


    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(features)

    dist, nbr_array = neigh.kneighbors(features, n_neighbors=k)



    return dist, nbr_array.tolist()


from joblib import Parallel, delayed
def calculate_weight(edge, features, alpha):
    node_indices = np.array(edge)
    data = features[node_indices].cpu().numpy()
    medians = np.median(data, axis=0)
    mad_distances = np.median(np.abs(data - medians), axis=0)
    dis_nor = np.mean(np.exp(-alpha * mad_distances))
    return dis_nor


def from_feature_kNN(args, features: torch.Tensor, k: int, device: torch.device = torch.device("cpu"), w_list: bool = True, mad: bool = True):
    r"""Construct the hypergraph from the feature matrix. Each hyperedge in the hypergraph is constructed by the central vertex ans its :math:`k-1` neighbor vertices.

    .. note::
        The constructed hypergraph is a k-uniform hypergraph. If the feature matrix has the size :math:`N \times C`, the number of vertices and hyperedges of the constructed hypergraph are both :math:`N`.

    Args:
        ``features`` (``torch.Tensor``): The feature matrix.
        ``k`` (``int``): The number of nearest neighbors.
        ``device`` (``torch.device``, optional): The device to store the hypergraph. Defaults to ``torch.device('cpu')``.
    """

    # 边的索引
    # dis_array, e_list = _e_list_from_feature_kNN(features.cpu(), k)
    dis_array, e_list = _e_list_from_feature_kNN11(features.cpu(), k)
    # tips MAD_distance
    if w_list and mad:

        # todo 加快训练速度
        w_list = []
        alpha = args.alpha
        # 将所有节点数据提前转换到 numpy 数组
        features_np = features.cpu().numpy()
        # 将 e_list 转换为 numpy 数组，方便批量处理
        e_array = np.array(e_list)
        # 批量提取所有节点的数据
        all_data = features_np[e_array]
        # 计算中位数和 MAD
        medians = np.median(all_data, axis=1)
        mad_distances = np.median(np.abs(all_data - medians[:, np.newaxis, :]), axis=1)
        # 将 MAD 转换为权重
        dis_nor = np.mean(np.exp(-alpha * mad_distances), axis=1)
        # 将结果转换为列表
        w_list = dis_nor.tolist()

        hg = Hypergraph(features.shape[0], e_list, w_list, device=device)

    if w_list and not mad:
    # # tips ED_distance
        w_list = []
        m_dist1 = sklearn.metrics.pairwise_distances(features.cpu())
        avg_dist = np.median(m_dist1)

        for i, edge in enumerate (e_list):
            node_indices = np.array(edge)
            data = features[node_indices].cpu()

            lower_triangle = sklearn.metrics.pairwise_distances(data)
            exp_term = np.mean(np.exp(-(lower_triangle ** 2 / avg_dist ** 2)))

            w_list.append(exp_term)

        hg = Hypergraph(features.shape[0], e_list, w_list, device=device)

    if not w_list:
        hg = Hypergraph(features.shape[0], e_list, device=device)

    return hg

def getDistanceToPro(x, pro):
    """
    obtain the distance to prototype for each instance
    Args:
        x: sample on the embedded space
    Returns: square of the euclidean distance, and the euclidean distance
    """

    xe = torch.unsqueeze(x, 1) - pro
    dist_to_centers = torch.sum(torch.mul(xe, xe), 2)
    euclidean_dist = torch.sqrt(dist_to_centers)

    return euclidean_dist.squeeze()

def get_err_threhold(fpr, tpr, threshold):
    differ_tpr_fpr_1=tpr+fpr-1.0
    right_index = np.argmin(np.abs(differ_tpr_fpr_1))
    best_th = threshold[right_index]
    dr = tpr[right_index]
    far = fpr[right_index]
    return dr, far, best_th, right_index


def convert_load(args, device, k_nebor, test_x, test_y,w_list,mad):

    # tensor
    test_x = torch.tensor(test_x).to(device).float()
    test_y = torch.tensor(test_y).to(device).long()

    if w_list == 't' and mad == 't':
        hg_test = from_feature_kNN(args,test_x, k=k_nebor,w_list=True, mad=True)
        hg_test = hg_test.to(device)

    elif w_list == 't'and mad == 'f':
        hg_test = from_feature_kNN(args,test_x, k=k_nebor,w_list=True, mad=False)
        hg_test = hg_test.to(device)

    elif w_list == 'f':
        hg_test = from_feature_kNN(args,test_x, k=k_nebor, w_list=False)
        hg_test = hg_test.to(device)

    return  test_x, test_y, hg_test


def generate_realistic_synthetic(X, y, realistic_synthetic_mode, alpha: int, percentage: float):
    '''
    Currently, four types of realistic synthetic outliers can be generated:
    1. local outliers: where normal data follows the GMM distribuion, and anomalies follow the GMM distribution with modified covariance
    2. global outliers: where normal data follows the GMM distribuion, and anomalies follow the uniform distribution
    3. dependency outliers: where normal data follows the vine coupula distribution, and anomalies follow the independent distribution captured by GaussianKDE
    4. cluster outliers: where normal data follows the GMM distribuion, and anomalies follow the GMM distribution with modified mean

    :param X: input X
    :param y: input y
    :param realistic_synthetic_mode: the type of generated outliers
    :param alpha: the scaling parameter for controling the generated local and cluster anomalies
    :param percentage: controling the generated global anomalies
    '''
    # if the dataset is too small, generating duplicate smaples up to n_samples_threshold
    # if len(y) < 1000:
    #     print(f'generating duplicate samples for dataset...')
    #     idx_duplicate = np.random.choice(np.arange(len(y)), 1000, replace=True)
    #     X = X[idx_duplicate]
    #     y = y[idx_duplicate]



    if realistic_synthetic_mode in ['local', 'cluster', 'dependency', 'global']:
        pass
    else:
        raise NotImplementedError

    # the number of normal data and anomalies
    pts_n = len(np.where(y == 0)[0])
    pts_a = len(np.where(y == 1)[0])

    # only use the normal data to fit the model
    X = X[y == 0]
    y = y[y == 0]

    # generate the synthetic normal data
    if realistic_synthetic_mode in ['local', 'cluster', 'global']:
        # select the best n_components based on the BIC value
        metric_list = []
        n_components_list = list(np.arange(1, 10))

        for n_components in n_components_list:
            gm = GaussianMixture(n_components=n_components, random_state=seed).fit(X)
            metric_list.append(gm.bic(X))

        best_n_components = n_components_list[np.argmin(metric_list)]

        # refit based on the best n_components
        gm = GaussianMixture(n_components=best_n_components, random_state=seed).fit(X)

        # generate the synthetic normal data
        X_synthetic_normal = gm.sample(pts_n)[0]

    # we found that copula function may occur error in some datasets
    elif realistic_synthetic_mode == 'dependency':
        # sampling the feature since copulas method may spend too long to fit
        if X.shape[1] > 50:
            idx = np.random.choice(np.arange(X.shape[1]), 50, replace=False)
            X = X[:, idx]

        copula = VineCopula('center')  # default is the C-vine copula
        copula.fit(pd.DataFrame(X))

        # sample to generate synthetic normal data
        X_synthetic_normal = copula.sample(pts_n).values

    else:
        pass

    # generate the synthetic abnormal data
    if realistic_synthetic_mode == 'local':
        # generate the synthetic anomalies (local outliers)
        gm.covariances_ = alpha * gm.covariances_
        X_synthetic_anomalies = gm.sample(pts_a)[0]

    elif realistic_synthetic_mode == 'cluster':
        # generate the clustering synthetic anomalies
        gm.means_ = alpha * gm.means_
        X_synthetic_anomalies = gm.sample(pts_a)[0]

    elif realistic_synthetic_mode == 'dependency':
        X_synthetic_anomalies = np.zeros((pts_a, X.shape[1]))

        # using the GuassianKDE for generating independent feature
        for i in range(X.shape[1]):
            kde = GaussianKDE()
            kde.fit(X[:, i])
            X_synthetic_anomalies[:, i] = kde.sample(pts_a)

    elif realistic_synthetic_mode == 'global':
        # generate the synthetic anomalies (global outliers)
        X_synthetic_anomalies = []

        for i in range(X_synthetic_normal.shape[1]):
            low = np.min(X_synthetic_normal[:, i]) * (1 + percentage)
            high = np.max(X_synthetic_normal[:, i]) * (1 + percentage)

            X_synthetic_anomalies.append(np.random.uniform(low=low, high=high, size=pts_a))

        X_synthetic_anomalies = np.array(X_synthetic_anomalies).T

    else:
        pass

    X = np.concatenate((X_synthetic_normal, X_synthetic_anomalies), axis=0)
    y = np.append(np.repeat(0, X_synthetic_normal.shape[0]),
                  np.repeat(1, X_synthetic_anomalies.shape[0]))

    return X, y


def tsne(data, dim):
    pca = sklearn.decomposition.PCA(n_components=dim)
    pca_result = pca.fit_transform(data)

    tsne = sklearn.manifold.TSNE(n_components=dim, perplexity=30, n_iter=1000, learning_rate='auto', random_state=20150101,
                                 init=pca_result)
    X_tsne = tsne.fit_transform(data)
    # Standardization
    scaler = sklearn.preprocessing.StandardScaler()
    X_norm = scaler.fit_transform(X_tsne)

    return X_norm




if __name__ == '__main__':
    # tips kdd数据处理  goad论文
    # url =  "../datasets/kddcup.data_10_percent (4).gz"
    urls = [
        "../datasets/kddcup.data_10_percent (4).gz",
        "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.names"
    ]

    df_colnames = pd.read_csv(urls[1], skiprows=1, sep=':', names=['f_names', 'f_types'])
    df_colnames.loc[df_colnames.shape[0]] = ['status', ' symbolic.']
    df = pd.read_csv(urls[0], header=None, names=df_colnames['f_names'].values)

    labels = np.where(df['status'] == 'normal.', 1, 0)

    # 删除 df 的最后一列
    df.drop(columns=df.columns[-1], inplace=True)

    df_symbolic = df_colnames[df_colnames['f_types'].str.contains('symbolic.')]
    # df_continuous = df_colnames[df_colnames['f_types'].str.contains('continuous.')]
    df_symbolic_names = df.columns.intersection(df_symbolic['f_names'])
    samples = pd.get_dummies(df.iloc[:, :-1], columns=df_symbolic_names)


    data =  samples.values
    label = labels

    combined_data = {'X': data.astype('float32'), 'y': label.astype('float32')}

    np.savez('../datasets/kddcup99.npz', **combined_data)

