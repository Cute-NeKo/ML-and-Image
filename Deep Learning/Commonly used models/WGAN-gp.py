import os
import numpy as np
import matplotlib.pyplot as plt
import copy
import seaborn
from tqdm import tqdm_notebook

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from unroll import mixture_gaussian
from torchvision import transforms

plt.style.use('ggplot')

device = torch.device("cuda:0")


def plot(points, title):
    plt.scatter(points[:, 0], points[:, 1], s=10, c='b', alpha=0.5)
    plt.scatter(dset.centers[:, 0], dset.centers[:, 1], s=100, c='g', alpha=0.5)
    plt.title(title)
    plt.ylim(-3, 3)
    plt.xlim(-3, 3)
    plt.show()
    plt.close()


def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(real_data.shape[0], 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)

    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.to(device)
    interpolates = interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size(), requires_grad=False).to(device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 1
    return gradient_penalty


dset = mixture_gaussian.data_generator()
sample_points = dset.sample(100)
plot(sample_points, 'Sampled data points')

# Model params (most of hyper-params follow the original paper: https://arxiv.org/abs/1611.02163)
z_dim = 2
g_inp = z_dim
g_hid = 512
g_out = dset.size

d_inp = g_out
d_hid = 512
d_out = 1

minibatch_size = 256

d_learning_rate = 1e-4
g_learning_rate = 1e-4
optim_betas = (0.5, 0.999)
log_interval = 150
d_steps = 5
g_steps = 1


def noise_sampler(N, z_dim):
    return np.random.normal(size=[N, z_dim]).astype('float32')


class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.activation_fn = nn.ReLU(True)

    def forward(self, x):
        x = self.activation_fn(self.map1(x))
        x = self.activation_fn(self.map2(x))
        return self.map3(x)


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.activation_fn = nn.ReLU(True)

    def forward(self, x):
        x = self.activation_fn(self.map1(x))
        x = self.activation_fn(self.map2(x))
        return self.map3(x)

    def load(self, backup):
        for m_from, m_to in zip(backup.modules(), self.modules()):
            if isinstance(m_to, nn.Linear):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()


def g_sample(G):
    with torch.no_grad():
        gen_input = torch.from_numpy(noise_sampler(minibatch_size, g_inp))
        gen_input = gen_input.to(device)
        g_fake_data = G(gen_input)
        return g_fake_data.cpu().numpy()


G = Generator(input_size=g_inp, hidden_size=g_hid, output_size=g_out)
D = Discriminator(input_size=d_inp, hidden_size=d_hid, output_size=d_out)
G.to(device)
D.to(device)
d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate)
g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate)

for e in range(20000):

    for z in range(4):
        d_real_data = torch.from_numpy(dset.sample(minibatch_size))
        d_real_data = d_real_data.to(device)
        d_fake_input = torch.from_numpy(noise_sampler(minibatch_size, g_inp))
        d_fake_input = d_fake_input.to(device)
        d_fake_data = G(d_fake_input)

        # train D
        d_optimizer.zero_grad()
        d_real_decision = D(d_real_data)

        d_fake_decision = D(d_fake_data)

        gradient_penalty = calc_gradient_penalty(D, d_real_data, d_fake_data)
        # print('g:',gradient_penalty)
        # print('m:',-torch.mean(d_real_decision) + torch.mean(d_fake_decision))

        loss_D = -torch.mean(d_real_decision) + torch.mean(d_fake_decision) + gradient_penalty
        loss_D.backward()
        d_optimizer.step()

    # train G
    g_optimizer.zero_grad()
    g_fake_input = torch.from_numpy(noise_sampler(minibatch_size, g_inp))
    g_fake_input = g_fake_input.to(device)
    g_fake_data = G(g_fake_input)
    dg_fake_decision = D(g_fake_data)
    loss_G = -torch.mean(dg_fake_decision)

    loss_G.backward()
    g_optimizer.step()

    samples = []
    if e % log_interval == 0:
        print('g:', gradient_penalty)
        g_fake_data = g_sample(G)
        samples.append(g_fake_data)
        plot(g_fake_data, title=' Iteration {}'.format(e))
        print(loss_D, loss_G)
