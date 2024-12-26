import torch
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn import manifold


def show_noise(args):
    print('Showing add_noise...')
    num_shows = 10
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    plt.rc('text', color='black')
    test_x = torch.tensor(args.test_x).float()
    # Generate images after adding noise at 10 step intervals up to 100 steps
    for i in range(num_shows):
        j = i // 5
        k = i % 5
        # x_i, _ = x_t(test_x, torch.tensor([i * args.num_steps // num_shows]), args)  # 生成t时刻的采样数据
        if i == 0:
            a = 0
            x_i, _ = x_t(test_x, torch.tensor([a]), args)
        elif i < 9:
            a = (i * args.num_steps // num_shows) + 10
            x_i, _ = x_t(test_x, torch.tensor([a]), args)
        else:
            a = 199
            x_i, _ = x_t(test_x, torch.tensor([a]), args)
        x_i = show_sample(x_i.cpu(), 2, 20150101)
        axs[j, k].scatter(x_i[:, 0], x_i[:, 1], color='lightsteelblue')
        # axs[j, k].set_axis_off()
        axs[j, k].set_xticks([])
        axs[j, k].set_yticks([])
        axs[j, k].set_title('$q(\mathbf{x}_{' + str(a) + '})$')

    # plt.xticks([])
    # plt.yticks([])
    plt.tight_layout(pad=1.08)
    plt.show()


def x_t(x_0, t, args):

    """It is possible to obtain x[t] at any moment t based on x[0]"""
    x_0 = x_0.to(args.device)
    noise = torch.randn_like(x_0).to(args.device)
    alphas_t = args.alphas_bar_sqrt[t].to(args.device)
    alphas_1_m_t = args.one_minus_alphas_bar_sqrt[t].to(args.device)
    # Add noise to x[0]
    return (alphas_t * x_0 + alphas_1_m_t * noise), noise

def show_sample(data,dimensions, rs):
    X = data
    pca = PCA(n_components=dimensions)
    pca_result = pca.fit_transform(X)
    tsne = manifold.TSNE(n_components=dimensions, perplexity=30, n_iter=1000, learning_rate='auto', random_state=rs, init=pca_result)
    X_tsne = tsne.fit_transform(X)

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)

    return X_norm