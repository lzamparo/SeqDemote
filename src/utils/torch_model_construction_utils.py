import torch
import torch.nn as nn


### FocalLoss (cf. arxiv:) to re-weight those w/ more uncertainty
class FocalLoss(nn.Module):
    
    def __init__(self, reduce=True, gamma=1.5, alpha=0.7):
        super(FocalLoss, self).__init__()
        self.reduce = reduce
        self.gamma = gamma
        self.alpha = alpha
        
    def _get_weights(self, x, t):
        '''
        Helper to get the weights for focal loss calculation
        '''
        p = nn.functional.sigmoid(x)
        p_t = p*t + (1 - p)*(1 - t)
        alpha_t = self.alpha * t + (1 - self.alpha)*(1 - t)
        w = alpha_t * (1 - p_t).pow(self.gamma)
        return w
    
    def focal_loss(self, x, t):
        '''
        Focal Loss cf. arXiv:1708.02002
        
        Args:
          x: (tensor) output from last layer of network
          t: (tensor) targets in [0,1]
          alpha: (float) class imbalance correction weight \in (0,1)
          gamma: (float) amplification factor for uncertain classification
          
        Return:
          (tensor) focal loss.
        '''
        weights = self._get_weights(x, t)
        return nn.functional.binary_cross_entropy_with_logits(x, t, weights, size_average=False, reduce=self.reduce)
    
    def forward(self, input, target):
        return self.focal_loss(input, target)


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


def get_sparse_weights_penalty(net, sparse_lambda=1e-6, cuda=True):
    ''' Return a list of additional sparsity penalties on those layers
    identfied in the model as sparse 
    '''
    sparse_penalties = []
    for name, p in net.named_parameters():
        if 'weight_v' in name:
            if cuda:
                p = p.cuda()
            L1_loss = sparse_lambda * (torch.abs(p)).sum()
            sparse_penalties.append(L1_loss)        
    return sparse_penalties


def orthogonal_filter_penalty(net, orth_lambda=1e-6, cuda=True):
    ''' Return a list of additional decorrelative penalty on the conv filters '''
    
    for name, p in net.named_parameters():
        if 'orth' in name and 'weight_v' in name:
            p_flattened = p.view(p.size(0),-1)
            WWt = torch.mm(p_flattened, torch.transpose(p_flattened,0,1))
            eye = torch.Tensor(torch.eye(p_flattened.size(0)))
            if cuda:
                eye = eye.cuda()
                WWt = WWt.cuda()
            WWt -= eye
            orth_loss = orth_lambda * WWt.sum()
    return [orth_loss]

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


 

    




