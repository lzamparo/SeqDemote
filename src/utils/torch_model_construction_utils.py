import torch
import torch.nn as nn

### Functions that allow for re-initialization of model and
### optimizer to tune hyperparameters 

def init_weights(m, gain=nn.init.calculate_gain('relu')):
    ''' Recursively initalizes the weights of a network. '''
    
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    
    if isinstance(m, nn.Conv2d):
        torch.nn.init.orthogonal_(m.weight, gain)
        m.bias.data.fill_(0.1)
    

def reinitialize_model(model, num_factors=19):
    ''' Initialize and return a model '''
    net = model(num_factors=num_factors)
    net.apply(init_weights)
    return net

def get_model_param_lists(net):
    ''' Partition the model parameters into biases, weights, and 
    weights intended to be sparse '''
    biases, weights, sparse_weights = [], [], []
    for name, p in net.named_parameters():
        if 'bias' in name:
            biases += [p]
            
        elif 'sparse' in name:
            sparse_weights += [p]
        
        else:
            weights += [p]  
            
    return biases, weights, sparse_weights

def orthogonal_filter_penalty(net, orth_lambda=1e-6, cuda=True):
    ''' Impose an additional decorrelative penalty on the conv filters '''
    
    for name, p in net.named_parameters():
        if 'orth' in name and 'weight_v' in name:
            p_flattened = p.view(p.size(0),-1)
            WWt = torch.mm(p_flattened, torch.transpose(p_flattened,0,1))
            print("type of tensor for WWt is: ", WWt.type())
            eye = torch.Tensor(torch.eye(p_flattened.size(0)))
            if cuda:
                print("Before .cuda(), type of tensor subtracted from WWt is: ", eye.type())
                eye = eye.cuda()
                WWt = WWt.cuda()
            print("type of tensor subtracted from WWt is: ", eye.type())
            print("type of tensor for WWt is: ", WWt.type())
            WWt -= eye
            orth_loss = orth_lambda * WWt.sum()
    return orth_loss

def initialize_optimizer(weights, biases, sparse_weights, hyperparams_dict):
    ''' Initialize the params, put together the arguments for the optimizer '''

    weight_lambda = hyperparams_dict['weight_lambda']
    bias_lambda = hyperparams_dict['bias_lambda']
    if sparse_weights:
        sparse_lambda = hyperparams_dict['sparse_lambda']
    
    optimizer = torch.optim.Adam
    optimizer_param_dicts = [
        {'params': weights, 'weight_decay': weight_lambda},
            {'params': biases, 'weight_decay': bias_lambda}
                        ]
    if sparse_weights:
        optimizer_param_dicts.append({'params': sparse_weights, 'weight_decay': sparse_lambda})
    return optimizer, optimizer_param_dicts

def get_sparse_weights_penalty(net, sparse_lambda=1e-6, cuda=True):
    ''' Impose an additional sparsity penalty on those layers
    identfied in the model as sparse '''
    sparse_penalties = []
    for name, p in net.named_parameters():
        if 'weight_v' in name:
            L1_loss = sparse_lambda * (torch.abs(p)).sum()
            sparse_penalties.append(L1_loss)        
    return sparse_penalties

 

    




