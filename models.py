import torch
from torch import nn

class Sequential_modified(nn.Sequential):
    def __init__(self, architecture, seed):
        self.seed = seed
        self.architecture = architecture
        torch.manual_seed(self.seed)
        self.architecture = architecture
        layer_sizes = list(zip(architecture[:-1],architecture[1:]))
        layer_list = []
        for size in layer_sizes[:-1]:
            layer_list.append(nn.Linear(*size))
            layer_list.append(nn.ReLU())
        layer_list.append(nn.Linear(*layer_sizes[-1]))
        super().__init__(*layer_list)

class Sequential_modified_2(nn.Sequential):
    def __init__(self, architecture, seed):
        self.seed = seed
        self.architecture = architecture
        torch.manual_seed(self.seed)
        self.architecture = architecture
        layer_sizes = list(zip(architecture[:-1],architecture[1:]))
        layer_list = []
        for size in layer_sizes[:-1]:
            layer_list.append(nn.Linear(*size))
            layer_list.append(nn.ReLU())
        layer_list.append(nn.Linear(*layer_sizes[-1]))
        super().__init__(*layer_list)

    def forward(self, input_):
        for layer in self:
            if isinstance(layer, nn.ReLU):
                in_shape = input_.shape[1]
                random_vector = torch.rand(in_shape)
                diag_mat = torch.diag(random_vector)
                diag_mat_inv = torch.diag(1/random_vector)
                input_ = torch.matmul(input_, diag_mat)
                input_ = layer(input_)
                input_ = torch.matmul(input_, diag_mat_inv)
            else:
                input_ = layer(input_)
        return input_