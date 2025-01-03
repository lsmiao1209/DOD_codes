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
    x_tensor = torch.tensor(x)

    if lam is None:
        lam = torch.rand(x_tensor.shape[0])  

    indices = torch.randperm(x_tensor.shape[0])  
    lam = lam.view(-1, 1)  
    new_x = x_tensor * (1 - lam) + x_tensor[indices] * lam 

    new_x = new_x.numpy()

    neg_y = np.ones(len(new_x))

    train_x = np.vstack((new_x, x))
    train_y = np.concatenate((neg_y, y))

    return train_x, train_y, neg_y



def SCARF_DA(x, y, corruption_rate=0.6):

    corruption_mask = np.random.rand(*x.shape) > corruption_rate

    min_vals = np.min(x, axis=0)
    max_vals = np.max(x, axis=0)

    x_random = min_vals + (max_vals - min_vals) * np.random.rand(*x.shape)

    x_corrupted = np.where(corruption_mask, x, x_random)


    neg_y = np.ones(len(x_corrupted))

    train_x = np.vstack((x_corrupted, x))
    train_y = np.concatenate((neg_y, y))

    return train_x,train_y,neg_y

def VAE_DA(X, y):
    from pyod.models.vae import VAE

    vae = VAE(encoder_neurons=[X.shape[1], 64, 128, 64, 32], decoder_neurons=[32, 64, 128, 64, X.shape[1]], latent_dim=16)
    vae.fit(X)


    latent_dim = 16
    n_samples_to_generate = X.shape[0]
    z_samples = np.random.normal(size=(n_samples_to_generate, latent_dim))

    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input


    decoder = vae.model_.get_layer('model_1')  



    generated_data = decoder.predict(z_samples)


    neg_y = np.ones(len(generated_data))

    train_x = np.vstack((generated_data, X))
    train_y = np.concatenate((neg_y, y))



    return train_x, train_y, neg_y
