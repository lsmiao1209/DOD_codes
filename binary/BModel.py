import torch
import torch.nn
from utils import *
from torch import nn
from torch.autograd import Function

class ModelP(nn.Module):
    def __init__(self, input_size, num_features, num_embedding, width, device):
        super(ModelP, self).__init__()
        # Number of instances  N
        self.input_size = input_size
        # Dimension of embedded feature spaces
        self.num_features = num_features
        self.num_embedding = num_embedding
        self.width = width
        self.trainmodel = True


        # # Encoder
        self.encoder = HGNNPE(self.num_features, self.width * self.num_features, self.num_embedding, use_bn=True).to(device)
        # Decoder
        self.decoder = HGNNPD(self.num_embedding, self.width * self.num_features, self.num_features, use_bn=True).to(device)

        self.fc = nn.Linear(self.width * num_features, num_embedding).to(device)

        self.fc1_update = nn.Linear(num_embedding, num_embedding).to(device)
        self.fc2_update = nn.Linear(num_embedding, num_embedding).to(device)

        self.fc1_reset = nn.Linear(num_embedding, num_embedding).to(device)
        self.fc2_reset = nn.Linear(num_embedding, num_embedding).to(device)

        self.fc1 = nn.Linear(num_embedding, num_embedding).to(device)
        self.fc2 = nn.Linear(num_embedding, num_embedding).to(device)

        self.attention_pooling = AttentionPooling(self.input_size)
        self.mean_pooling = MeanPolling()

        # Discriminator
        self.discrimtor = HGNNPDis(self.num_embedding, self.width * self.num_embedding, 2, use_bn=True).to(device)


    def forward(self, X, hg):

        x_e, x_e1 = self.encoder(X, hg)
        x_e1 = self.fc(x_e1)

        z = torch.sigmoid(self.fc1_update(x_e1) + self.fc2_update(x_e))
        r = torch.sigmoid(self.fc1_reset(x_e1) + self.fc2_reset(x_e))
        out = torch.tanh(self.fc1(x_e1) + self.fc2(r * x_e))
        outs = z * out + (1-z) * x_e


        proto = self.attention_pooling(outs)

        # -----else the discriminator predicts the subgroup assignment for each instance----- #
        reversed_x_e = GradientReversalLayer.apply(outs)
        xdis = self.discrimtor(reversed_x_e,hg)

        if self.trainmodel:
            x_de = self.decoder(outs, hg)
            return outs, x_de, proto, xdis

        x_de = self.decoder(outs, hg)

        return outs, x_de

class GradientReversalLayer(Function):

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg(), None
