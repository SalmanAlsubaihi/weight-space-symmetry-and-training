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