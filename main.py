import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from utils import *
from torch.utils.tensorboard import SummaryWriter
import itertools
import shutil
from torch.nn.utils import clip_grad_norm_
# from other_training_functions import *

hparams_all_exp = {
    'architecture' : [[784,100,100,10,10], [784,100,100,100,100,100,100,10]],
    'lr' : [0.001,0.1,0.01],
    'num_epochs' : [20],
    'seed' : [200],
    'weight_symmetry_parameters':[{'name':'diagonal_matrix_symmetry_balanced_3', 'parameters':{'scale':100}}, None],
    'data_parameters':[{'name' : 'random_data', 'parameters':{'num_iterations' : 2000, 'batch_size' : 1000, 'SGD' : False}}],
}
# {'name':'diagonal_matrix_symmetry_balanced_3', 'parameters':{'scale':100}}, 
criterion_list = {
    'random_data': nn.MSELoss,
    'mnist' : nn.CrossEntropyLoss
}

# 'weight_symmetry_parameters':{'name':'diagonal_matrix_symmetry_1', 'parameters':{'range_':[0.5,1.5]}}

symmetry_functions = {
    'scalar_symmetry':scalar_symmetry,
    'diagonal_matrix_symmetry_1':diagonal_matrix_symmetry_1,
    'diagonal_matrix_symmetry_balanced_1':diagonal_matrix_symmetry_balanced_1,
    'diagonal_matrix_symmetry_balanced_2':diagonal_matrix_symmetry_balanced_2,
    'diagonal_matrix_symmetry_balanced_3':diagonal_matrix_symmetry_balanced_3
}

def train(
    hparams,
    exp_name,
    module,
    data_loader,
    optim,
    criterion,
    num_epochs,
    num_iterations = int,
    stationary_point_condition = module_at_stationary_point_2,
    revert_condition = revert_condition,
    lr = None,
    architecture = None,
    weight_symmetry_parameters = None,
    **kwargs):
    
    log_path = 'runs/'+exp_name
    shutil.rmtree(log_path, ignore_errors=True)
    writer = SummaryWriter(log_dir=log_path)
    check_interval = 100
    # writer.add_hparams(hparams)
    iter_loss = []
    old_net = None
    device = module[0].weight.device
    symm_counter = 0
    iter_ = -1
    for epoch in range(num_epochs):
        if epoch >= 1:
            iter_list, loss_list = zip(*iter_loss)
            print(f'Epoch: {epoch}    Loss= {torch.tensor(loss_list[-100:]).mean()}')
        for x, target in data_loader:
            iter_ += 1
            x, target = x.to(device), target.to(device)
            optim.zero_grad()
            output = module(x)
            loss = criterion(output,target)

            if weight_symmetry_parameters and (iter_+1)%20 == 0 and revert_condition(loss, iter_loss):  # make "if weight_symmetry_parameters" over both if conditions
                print(f'[{iter_}] Bad location:  retrive to old module ..   loss= {loss.item()}')
                if old_net:
                    copy_weights(old_net, module)
                    optim = torch.optim.SGD(module.parameters(), lr=0.001) #delete
                    symm_counter += 1 #should I keep this line??
                    continue

            loss.backward()

            norm = clip_grad_norm_(module.parameters(), 10)
            # if iter_%100 == 0:
            #     print(f'grad norm = {norm}')
            optim.step()
            iter_loss.append((iter_,loss.item()))
            if iter_%10 == 0:
                writer.add_scalar('loss', loss.item(), iter_)

            if weight_symmetry_parameters and (iter_+1)%check_interval == 0 and stationary_point_condition(iter_loss, iter_interval=check_interval//2): ## add 100 as hparams , also link to iter_interval in module_at_stationary_point
                print(f'[{iter_}] At stationary point:  calculating eqivelant module ..')
                if not old_net:
                    old_net = Sequential_modified(module.architecture, random.randint(0,10000)) #TODO: fix
                    old_net = old_net.to(device)

                if symm_counter%2 == 0: #???????
                    copy_weights(module, old_net)
        #         factors = generate_numbers_mul_to_1(len(width_list)-1,[0.7n = clone_net(module),1.3])
        #         module = scale_net(module, factors)
                symmetry_function = symmetry_functions[weight_symmetry_parameters['name']]
                symmetry_parameters = weight_symmetry_parameters['parameters']
                module = symmetry_function(module, symm_counter, **symmetry_parameters)
                symm_counter += 1

        if symm_counter%2 == 1:
            module = symmetry_function(module, symm_counter, **symmetry_parameters)
            symm_counter += 1

        weights = get_net_weights_as_one_tensor(module)
        writer.add_histogram('All Layers Histogram', weights, epoch)


def run_exp(
    hparams,
    device,
    exp_name,
    architecture = None,
    lr = None,
    seed = None,
    data_parameters = None,
    **kwargs): ####  can I remove None??
    
    print(f'Running:\n{exp_name}\n\n')
    data_loader = get_data_loader(data_parameters['name'], architecture, **data_parameters['parameters'])
    criterion = criterion_list[data_parameters['name']]()


    module = Sequential_modified(architecture, seed).to(device)
    optim = torch.optim.SGD(module.parameters(), lr=lr)

    train(hparams, exp_name, module, data_loader, optim, criterion, **kwargs)
    # conservative_training(hparams, exp_name, module, data_loader, optim, criterion, **kwargs)

def main(hparams_all_exp):
    device = 'cuda:1'
    prod = itertools.product(*list(hparams_all_exp.values()))
    hparams_list = []
    for elem in prod:
        hp = dict(zip(hparams_all_exp.keys(), elem))
        hparams_list.append(hp)
    for hparams in hparams_list:
        exp_name = ' '.join([f'{k}={v["name"]}' if isinstance(v,dict) else f'{k}={v}' for k,v in hparams.items()])
        run_exp(hparams, device, exp_name, **hparams)





def conservative_training(
    hparams,
    exp_name,
    module,
    data_loader,
    optim,
    criterion,
    num_epochs,
    num_iterations = int,
    stationary_point_condition = module_at_stationary_point_2,
    revert_condition = revert_condition,
    lr = None,
    architecture = None,
    weight_symmetry_parameters = None,
    **kwargs):
    
    log_path = 'runs/'+exp_name
    shutil.rmtree(log_path, ignore_errors=True)
    writer = SummaryWriter(log_dir=log_path)

    symmetry_function = symmetry_functions[weight_symmetry_parameters['name']]
    symmetry_parameters = weight_symmetry_parameters['parameters']
    # writer.add_hparams(hparams)
    iter_loss = []
    old_net = None
    device = module[0].weight.device
    symm_counter = 0
    iter_ = -1
    flag=0
    for epoch in range(num_epochs):
        if epoch >= 1:
            iter_list, loss_list = zip(*iter_loss)
            print(f'Epoch: {epoch}    Loss= {torch.tensor(loss_list[-100:]).mean()}')
        for x, target in data_loader:
            iter_ += 1
            x, target = x.to(device), target.to(device)
            optim.zero_grad()
            output = module(x)
            loss = criterion(output,target)
            loss.backward()
            norm = clip_grad_norm_(module.parameters(), 0.1)
            optim.step()
            iter_loss.append((iter_,loss.item()))

            if flag == 1:
                module = symmetry_function(module, symm_counter, **symmetry_parameters)
                symm_counter += 1
                flag = 0

            if iter_%10 == 0:
                writer.add_scalar('loss', loss.item(), iter_)

            if weight_symmetry_parameters and (iter_+1)%100 == 0 and stationary_point_condition(iter_loss): ## add 100 as hparams , also link to iter_interval in module_at_stationary_point
                print(f'[{iter_}] At stationary point:  calculating eqivelant module ..')
                module = symmetry_function(module, symm_counter, **symmetry_parameters)
                symm_counter += 1
                flag = 1
        weights = get_net_weights_as_one_tensor(module)
        writer.add_histogram('All Layers Histogram', weights, epoch)



if __name__ == "__main__":
    main(hparams_all_exp)