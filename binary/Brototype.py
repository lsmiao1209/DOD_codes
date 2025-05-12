import time
import torch.nn
import torch.optim as optim
from torch import nn

import Tools.data_augmentation
from binary.BModel import ModelP
from Tools.utils import *
from torchsummary import summary
from train_val import train

class AttentionPooling(nn.Module):
    def __init__(self, size):
        super(AttentionPooling, self).__init__()
        self.n = size -1
        self.sigmoid = nn.Sigmoid()


    def forward(self, outs):
        # Calculate attention scores
        n = self.n
        d = (outs - outs.mean(dim=0)).pow(2)
        v = d.sum(dim=0) / n
        e = d / (4 * (v + 0.001)) + 0.5
        proto = torch.sum(outs * self.sigmoid(e.sum(dim=1)).unsqueeze(1), dim=0)/ self.sigmoid(e.sum(dim=1)).sum()

        return proto

class MeanPolling(nn.Module):
    def __init__(self):
        super(MeanPolling, self).__init__()

    def forward(self, x):
        proto = torch.mean(x, dim=0)
        return proto


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

class AutomaticWeightedLoss(torch.nn.Module):
    def __init__(self, num=3):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


def BPROT(dataname, device, train_x, train_y, test_x, test_y, args):
    st = time.time()
    if args.t == 0:
        train_x, train_y = train_x, train_y


    elif args.t == 1000:
        train_x, train_y, _ = Tools.data_augmentation.diffusion(args.t, "linear", train_x, train_y)

    # weight: t/f
    w_list = args.w_list
    mad = args.mad
    num_embedding = args.num_embedding
    # neighbors
    k_nebor = args.k_nebor
    num_features = train_x.shape[1]

    # convert
    train_x, train_y, test_x, test_y, hg_train, hg_test,_,_ = convert(args, device, k_nebor, train_x, train_y, test_x, test_y, w_list,mad)
    timetaken = time.time() - st
    # print('hypergraph time:', timetaken)

    st = time.time()
    # Initial
    net = ModelP(train_x.shape[0], num_features, num_embedding, args.width, device)
    optimizer = opt(net)
    mse = torch.nn.MSELoss()
    crossloss = torch.nn.CrossEntropyLoss(label_smoothing=0.0)

    loss_module = AutomaticWeightedLoss(num=3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

    attention_pooling = AttentionPooling(train_x.shape[0])
    # attention_pooling = MeanPolling()


    auclist, prlist, timetaken = train(dataname, train_x, train_y, test_x, test_y, hg_train, hg_test,num_embedding,num_features,device, args/
                                       net,optimizer,mse,crossloss,loss_module,scheduler,attention_pooling)

    return auclist, prlist, timetaken

def opt(net):
    opt = optim.Adam([
        {'params': net.encoder.parameters(),'lr':0.001},
        {'params': net.decoder.parameters(),'lr':0.001},
        {'params': net.fc.parameters()},
        {'params': net.fc1.parameters()},
        {'params': net.fc2.parameters()},
        {'params': net.fc1_update.parameters()},
        {'params': net.fc1_reset.parameters()},
        {'params': net.fc2_reset.parameters()},
        {'params': net.fc2_update.parameters()},
        {'params': net.discrimtor.parameters(), 'lr': 0.00001}
    ], lr=0.01)
    return opt

