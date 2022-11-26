import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torch.nn.utils import clip_grad_norm_


def generate_numbers_mul_to_1(num, range_):
    torch.manual_seed(random.randint(0,10000))
    rand_numbers = (range_[1]-range_[0]) * torch.rand(num) + range_[0]
    m = 1.0
    for i in range(num):
        m *= rand_numbers[i]
    rand_numbers = rand_numbers/(m**(1/num))
    return rand_numbers

def scalar_symmetry(net, *args, range_=None):
    layers = list(filter(lambda layer:isinstance(layer, nn.Linear), net))
    factors = generate_numbers_mul_to_1(len(layers), range_)
    for i, layer in enumerate(layers):
        layer.weight.data = factors[i] * layer.weight.data
        layer.bias.data = multiply_list_of_num(factors[:i+1]) * layer.bias.data
    return net

def diagonal_matrix_symmetry_1(net, *args, range_=None):
    device = net[0].weight.device
    for i, layer in enumerate(net):
        if isinstance(layer, nn.ReLU):
            d = net[i-1].weight.data.shape[0]
            v = (range_[1]-range_[0])*torch.rand(d) + range_[0]
            diag_mat = torch.diag(v).to(device)
            diag_mat_inv = torch.diag(1/v).to(device)
            net[i-1].weight.data = torch.matmul(diag_mat, net[i-1].weight.data)
            net[i-1].bias.data = torch.matmul(diag_mat, net[i-1].bias.data)
            net[i+1].weight.data = torch.matmul(net[i+1].weight.data, diag_mat_inv)
    return net


def diagonal_matrix_symmetry_balanced_1(net, symm_counter, range_):
    device = net[0].weight.device
    torch.manual_seed(symm_counter//2)
    for i, layer in enumerate(net):
        if isinstance(layer, nn.ReLU):
            d = net[i-1].weight.data.shape[0]
            v = (range_[1]-range_[0])*torch.rand(d) + range_[0]
            if symm_counter%2 == 1:
                v = 1/v
            diag_mat = torch.diag(v).to(device)
            diag_mat_inv = torch.diag(1/v).to(device)
            net[i-1].weight.data = torch.matmul(diag_mat, net[i-1].weight.data)
            net[i-1].bias.data = torch.matmul(diag_mat, net[i-1].bias.data)
            net[i+1].weight.data = torch.matmul(net[i+1].weight.data, diag_mat_inv)
    return net


def diagonal_matrix_symmetry_balanced_2(net, symm_counter, scale):
    device = net[0].weight.device
    torch.manual_seed(symm_counter//2)
    for i, layer in enumerate(net):
        if isinstance(layer, nn.ReLU):
            d = net[i-1].weight.data.shape[0]
            v = (2*torch.rand(d) - 1)
            v = ((v>0)*v) * scale + ((v<0)*v.abs()) / scale
            if symm_counter%2 == 1:
                v = 1/v
            diag_mat = torch.diag(v).to(device)
            diag_mat_inv = torch.diag(1/v).to(device)
            net[i-1].weight.data = torch.matmul(diag_mat, net[i-1].weight.data)
            net[i-1].bias.data = torch.matmul(diag_mat, net[i-1].bias.data)
            net[i+1].weight.data = torch.matmul(net[i+1].weight.data, diag_mat_inv)
    return net


def diagonal_matrix_symmetry_balanced_3(net, symm_counter, scale):
    device = net[0].weight.device
    torch.manual_seed(symm_counter//2)
    for i, layer in enumerate(net):
        if isinstance(layer, nn.ReLU):
            d = net[i-1].weight.data.shape[0]
            v = (2*torch.rand(d) - 1)
            v = ((v>0)*v) * scale + ((v<0)*(v.abs()+(1/scale)))
            if symm_counter%2 == 1:
                v = 1/v
            diag_mat = torch.diag(v).to(device)
            diag_mat_inv = torch.diag(1/v).to(device)
            net[i-1].weight.data = torch.matmul(diag_mat, net[i-1].weight.data)
            net[i-1].bias.data = torch.matmul(diag_mat, net[i-1].bias.data)
            net[i+1].weight.data = torch.matmul(net[i+1].weight.data, diag_mat_inv)
    return net


def turn_grad_off(net, weights=True, bias=True):
    for idx, param in net.named_children():
        if isinstance(param, nn.Linear):
            if weights:
                param.weight.requires_grad = False
            if bias:
                param.bias.requires_grad = False
    return net

def diff(net1, net2):
    input_size = net1[0].weight.data.shape[1]
    random_input = torch.randn(10000,input_size)
    error = (net2(random_input)-net1(random_input)).abs().max()
    return error

def multiply_list_of_num(list_):
    multiplication = 1
    for num in list_:
        multiplication *= num
    return multiplication


def module_at_stationary_point_1(iter_loss, iter_interval=100, percentage=1):
    if len(iter_loss) > iter_interval:
        loss1, loss2 = iter_loss[-iter_interval][1], iter_loss[-1][1]
        return abs(loss1-loss2)/loss1*100 < percentage
    return False

def module_at_stationary_point_2(iter_loss, iter_interval=100, percentage=0.5):
    if len(iter_loss) > iter_interval:
        iter_, loss_list = zip(*iter_loss)
        w = iter_interval//2
        loss1, loss2 = torch.tensor(loss_list[-iter_interval:-w]).mean(), torch.tensor(loss_list[-w:]).mean()
        return abs(loss1-loss2)/loss1*100 < percentage
    return False

def module_at_stationary_point_3(module, min_norm=0.05):
    norm = clip_grad_norm_(module.parameters(), 100000)
    return norm < min_norm

def copy_weights(src_net, dst_net):
    with torch.no_grad():
        for dst_params, src_params in zip(dst_net.parameters(), src_net.parameters()):
            dst_params.copy_(src_params)

class Random_data():
    def __init__(self, Architecture, batch_size, num_iterations):
        self.x = torch.randn(batch_size, Architecture[0])
        labels_net = Sequential_modified(Architecture, 123) # Should seed be random
        self.targets = labels_net(self.x).detach()
        self.num_iterations = num_iterations
    def __getitem__(self, idx):
        return self.x, self.targets
    def __len__(self):
        return self.num_iterations

def revert_condition(loss, iter_loss, threshold=0.1, interval=20):
    if len(iter_loss) > interval:
        iter_, loss_list = zip(*iter_loss)
        w = interval//2
        loss_1 = torch.tensor(loss_list[-interval:-w]).mean()
        loss_2 = torch.tensor(loss_list[-w:]).mean()
        # return (loss_2-loss_1)/loss_1 > threshold
        return iter_loss[-1][0] > 1 and (loss-iter_loss[-interval][1])/iter_loss[-interval][1] > threshold
    return False

def squeeze(tens):
    return tens.squeeze(0).squeeze(0)


def get_data_loader(data_name, architecture, num_iterations, batch_size, **kwargs):
    if data_name == 'random_data':
        data = Random_data(architecture, batch_size, num_iterations=num_iterations)
        data_loader = DataLoader(data)
    elif data_name == 'mnist':
        transform = transforms.Compose(
        [transforms.Resize((1,28*28)),
         transforms.ToTensor(),
         squeeze
        ])
        train_data = torchvision.datasets.MNIST('datasets/', transform=transform, download=True)
        data_loader = DataLoader(train_data, batch_size=batch_size)
    else:
        raise Exception('Unsupported Dataset!')
    return data_loader

def get_net_weights_as_one_tensor(net):
    tensors = []
    for layer in net:
        if isinstance(layer,nn.Linear) or isinstance(layer,nn.Conv2d):
            tensors.append(layer.weight.data.flatten())
            tensors.append(layer.bias.data.flatten())
    all_weights = torch.cat(tensors)
    return all_weights