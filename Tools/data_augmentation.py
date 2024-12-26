import argparse
import time

import sklearn.manifold
import torch
import matplotlib.pyplot as plt
import numpy as np
import sklearn.impute

import Tools.utils
from tabgan.sampler import OriginalGenerator, GANGenerator, ForestDiffusionGenerator, LLMGenerator
import pandas as pd
import numpy as np

def evaluate_similarity(train_x, neg_x):
    from sdv.metadata import SingleTableMetadata
    from sdv.evaluation.single_table import evaluate_quality

    # 确定列数并生成默认列名
    num_columns = train_x.shape[1]
    column_names = [f'column{i + 1}' for i in range(num_columns)]

    # 转换为 Pandas DataFrame
    real_data = pd.DataFrame(train_x, columns=column_names)
    synthetic_data = pd.DataFrame(neg_x, columns=column_names)

    # 创建元数据对象
    metadata = SingleTableMetadata()
    # 假设所有字段为数值型
    for column in column_names:
        metadata.add_column(column, sdtype='numerical')

    # 评估合成数据的质量
    quality_report = evaluate_quality(
        real_data,
        synthetic_data,
        metadata
    )


def x_t(x_0, t, args):

    """It is possible to obtain x[t] at any moment t based on x[0]"""
    noise = np.random.randn(*x_0.shape)
    alphas_t = args.alphas_bar_sqrt[t]
    alphas_1_m_t = args.one_minus_alphas_bar_sqrt[t]
    # Add noise to x[0]
    return (alphas_t * x_0 + alphas_1_m_t * noise), noise

def diffusion(steps, schedule_name, train_x, train_y):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.num_steps = steps
    args.betas = get_named_beta_schedule(schedule_name, args.num_steps)

    args.alphas = 1 - args.betas
    args.alphas_bar = np.cumprod(args.alphas, 0)
    args.alphas_bar_sqrt = np.sqrt(args.alphas_bar)
    args.one_minus_alphas_bar_sqrt = np.sqrt(1 - args.alphas_bar)

    # Generate a NumPy array of random integers from 0 to T.
    t = np.random.randint(0, args.num_steps, size=(train_x.shape[0],)).reshape(-1, 1)


    # Constructing inputs to the model
    neg_x, noise = x_t(train_x, t, args)

    # tips 评估分数
    # evaluate_similarity(train_x, neg_x)


    neg_y = np.ones(len(neg_x))


    train_x = np.vstack((neg_x, train_x))
    train_y = np.concatenate((neg_y, train_y))

    train_x, train_y = Tools.utils.shuffle(train_x, train_y)

    # tsne(train_x, train_y, 2)



    return train_x, train_y, neg_x


def double(train_x, train_y):


    # Constructing inputs to the model
    neg_x = train_x

    # tips 评估分数
    # evaluate_similarity(train_x, neg_x)


    neg_y = np.ones(len(neg_x))

    # Tools.utils.draw_synthetic("generate_samples", neg_x, train_x)

    # hot01(neg_x, train_x)


    train_x = np.vstack((neg_x, train_x))
    train_y = np.concatenate((neg_y, train_y))

    train_x, train_y = Tools.utils.shuffle(train_x, train_y)

    return train_x, train_y,neg_x


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        scale = 1e-6
        beta_start = scale * 1e-3
        beta_end = scale * 2e-2
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def CTGAN_DA(train, target):

    train_df = pd.DataFrame(train)
    new_train2, new_target2 = GANGenerator(
        gen_params={"batch_size": 500, "epochs": 10, "patience": 5}, pregeneration_frac=1, only_generated_data=True).generate_data_pipe(train_df, target,None,only_generated_data=True, )

    neg_x = new_train2.values
    neg_y = np.ones(len(neg_x))

    # evaluate_similarity(train, neg_x)

    train_x = np.vstack((neg_x, train))
    train_y = np.concatenate((neg_y, target))
    return train_x,train_y,neg_y



def MIXUP_DA(x, y,lam=None, target_reweighted=None):
    """
    Args:
        x (tensor): 输入，形状为 (N, M)
        lam (tensor): 混合系数，形状为 (N,)
        target_reweighted (tensor): 目标标签，形状为 (N, num_classes)

    Returns:
        new_x (tensor): 输出，形状为 (N, M)，表示混合后的输入数据
        target_reweighted (tensor): 混合后的目标标签，形状为 (N, num_classes)
    """
    x_tensor = torch.tensor(x)

    if lam is None:
        lam = torch.rand(x_tensor.shape[0])  # 生成一个随机的混合系数向量 lam，形状为 (N,)

    indices = torch.randperm(x_tensor.shape[0])  # 在相同设备上生成随机索引
    lam = lam.view(-1, 1)  # 将 lam 变为列向量以便进行广播
    new_x = x_tensor * (1 - lam) + x_tensor[indices] * lam  # 对每个样本应用混合系数

    new_x = new_x.numpy()

    neg_y = np.ones(len(new_x))

    train_x = np.vstack((new_x, x))
    train_y = np.concatenate((neg_y, y))

    return train_x, train_y, neg_y

    # if target_reweighted is not None:
    #     target_shuffled_onehot = target_reweighted[indices]
    #     target_reweighted = target_reweighted * (1 - lam) + target_shuffled_onehot * lam
    #     return new_x, target_reweighted
    # else:
    #     return new_x, None


def SCARF_DA(x, y, corruption_rate=0.6):
    """
    Args:
        x (np.ndarray): 输入，形状为 (N, M)
        corruption_rate (float): 遮盖率

    Returns:
        x_corrupted (np.ndarray): 输出，形状为 (N, M)，表示被遮盖后的输入数据
    """
    # 生成遮盖矩阵
    corruption_mask = np.random.rand(*x.shape) > corruption_rate

    # 生成每个特征的最小值和最大值
    min_vals = np.min(x, axis=0)
    max_vals = np.max(x, axis=0)

    # 从每个特征的实际范围中采样随机值
    x_random = min_vals + (max_vals - min_vals) * np.random.rand(*x.shape)

    # 使用遮盖矩阵进行数据替换
    x_corrupted = np.where(corruption_mask, x, x_random)

    # evaluate_similarity(x, x_corrupted)

    neg_y = np.ones(len(x_corrupted))

    train_x = np.vstack((x_corrupted, x))
    train_y = np.concatenate((neg_y, y))

    return train_x,train_y,neg_y

# tips 画散点图，生成前后
def tsne(data, y, dim):
    pca = sklearn.decomposition.PCA(n_components=dim)
    pca_result = pca.fit_transform(data)

    tsne = sklearn.manifold.TSNE(n_components=dim, perplexity=30, n_iter=1000, learning_rate='auto', random_state=20150101,
                                 init=pca_result)
    X_tsne = tsne.fit_transform(data)
    # Standardization
    scaler = sklearn.preprocessing.StandardScaler()
    X_norm = scaler.fit_transform(X_tsne)


    normal_original_data = X_norm[y == 0]
    anomaly_original_data = X_norm[y == 1]

    # 绘制数据
    plt.figure(figsize=(10, 6))

    # 绘制原始数据
    plt.scatter(normal_original_data[:, 0], normal_original_data[:, 1], color='blue', alpha=0.6, label='Original Data')

    # 绘制生成数据
    plt.scatter(anomaly_original_data[:, 0], anomaly_original_data[:, 1], color='red', alpha=0.6, label='Generated Data')

    # 添加标签和标题
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Original vs. Generated Data')
    plt.legend()

    # 显示图像
    plt.show()

    return X_norm

def VAE_DA(X, y):
    from pyod.models.vae import VAE
    # 创建并训练VAE模型
    vae = VAE(encoder_neurons=[X.shape[1], 64, 128, 64, 32], decoder_neurons=[32, 64, 128, 64, X.shape[1]], latent_dim=16)
    vae.fit(X)

    # 从潜在空间中采样
    latent_dim = 16
    n_samples_to_generate = X.shape[0]
    z_samples = np.random.normal(size=(n_samples_to_generate, latent_dim))

    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input

    # 提取解码器部分
    decoder = vae.model_.get_layer('model_1')  # 解码器
    # decoder_input = Input(shape=(vae.latent_dim,))  # 创建解码器输入层
    # decoder_output = decoder(decoder_input)  # 获取解码器输出
    # decoder_model = Model(inputs=decoder_input, outputs=decoder_output)

    # 生成数据
    generated_data = decoder.predict(z_samples)

    # evaluate_similarity(X, generated_data)

    neg_y = np.ones(len(generated_data))

    train_x = np.vstack((generated_data, X))
    train_y = np.concatenate((neg_y, y))

    # a = tsne(train_x, train_y, 2)

    return train_x, train_y, neg_y