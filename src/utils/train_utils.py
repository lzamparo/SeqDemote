import numpy as np 
import gzip
import os
from collections import OrderedDict
from sklearn.metrics import roc_auc_score, average_precision_score


### Weak AggMo implementation attempt.   
def apply_aggregated_threevec_momentum(updates, params=None, velocity_updates=None, momentums=np.array([0.,0.9,0.999])):
    """Returns a modified update dictionary including aggregated momentum
    
    Generates update expressions of the form:
    * ``velocity_i := momentum_i * velocity_(i-i) + updates[param] - param``
    * ``param := param + lr*velocity``
    Parameters
    ----------
    updates : OrderedDict
        A dictionary mapping parameters to update expressions
    params : iterable of shared variables, optional
        The variables to apply momentum to. If omitted, will apply
        momentum to all `updates.keys()`.
    momentums : numpy array of floats or symbolic scalars, optional
        The amount of momentum to apply to each velocity vector
        Higher momentum results in smoothing over more update steps, 
        and averaging multiple vectors results in more stable convergence
        (cf. https://arxiv.org/abs/1804.00325)
        Defaults to np.array([0. 0.9, 0.999])
    
    Returns
    -------
    OrderedDict
        A copy of `updates` with momentum updates for all `params`.
    Notes
    -----
    Higher momentum also results in larger update steps. To counter that,
    you can optionally scale your learning rate by `1 - momentum`.
    See Also
    --------
    momentum : Shortcut applying momentum to SGD updates
    """
    if params is None:
        params = updates.keys()
    updates = OrderedDict(updates)
    
    if velocity_updates is None:
        velocity_updates = OrderedDict()
        for param in params:
            value = param.get_value(borrow=True)
            velocity_updates[param] = [theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                 broadcastable=param.broadcastable) for i in range(momentums.shape[0])]

    # calculate update expressions, using velocity_updates to update each velocity vector 
    # prior to calculating the update for each model parameter
    for param in params:
        
        # calculate each velocity update
        velocity_updates[param] = [momentum * velocity + updates[param] for momentum, velocity in zip(
            velocity_updates[param], momentums)]
        
        # calculate parameter update
        #x = m * velocities + updates[param] 
        K = momentums.shape[0]
        updates[param] = updates[param] + (1. / K)  * sum(velocity_updates[param])

    
    return updates, velocity_updates


def per_task_loss(y_hat, y, loss, do_sum=True):
    ''' Calculate the per-task loss.  Shape of y, y_hat assumed 
    to be like (samples, tasks) '''
    
    all_task_losses = [loss(y_hat[:,c], y[:,c]) for c in range(y_hat.shape[1])]
    if do_sum:
        return sum(all_task_losses)
    else:
        return all_task_losses

### Log-loss calculating utils

def one_hot(vec, m=None):
    if m is None:
        m = int(np.max(vec)) + 1

    return np.eye(m)[vec]

def log_loss_std(y, t, eps=1e-15):
    """
    cross entropy loss, summed over classes, mean over batches
    """
    losses = log_losses(y, t, eps)
    return np.std(losses)


def log_losses(y, t, eps=1e-15):
    if t.ndim == 1:
        t = one_hot(t)

    y = np.clip(y, eps, 1 - eps)
    losses = -np.sum(t * np.log(y), axis=1)
    return losses

def log_loss(y, t, eps=1e-15):
    """
    cross entropy loss, summed over classes, mean over batches
    """
    losses = log_losses(y, t, eps)
    return np.mean(losses)

def st_accuracy(y, y_hat):
    """ single-task (peak vs flank) ROC ;
    y_hat := predicted labels
    y := actual labels from data """
    if not y_hat.shape == y.shape:
        print("Error in single task AUC: shape mismatch in \hat{y}: ", y_hat.shape, " and y: ", y.shape)
        return -1
    return roc_auc_score(y, y_hat)

def mt_accuracy(y, y_hat):
    """ 
    multi-task ROC: the un-weighted average of task ROC scores; 
    y_hat := predicted labels
    y := actual labels from data
    """
    if not y_hat.shape == y.shape:
        print("Error in AUC: shape mismatch for \hat{y}: ", y_hat.shape, " and y: ", y.shape)
        return -1
    rocs = []
    for targets, preds in zip(y.transpose(), y_hat.transpose()):
        rocs.append(roc_auc_score(targets, preds))
    
    return np.mean(rocs)

def mt_precision(y, y_hat):
    """
    multi-task precision: the un-weighted average of task precision scores;
    y_hat := predicted labels
    y := actual labels from data
    """
    if not y_hat.shape == y.shape:
        print("Error in precision: shape mismatch for \hat{y}: ", y_hat.shape, " and y: ", y.shape)
        return -1
    precisions = []
    for targets, preds in zip(y.transpose(), y_hat.transpose()):
        precisions.append(average_precision_score(targets, preds))
    
    return np.mean(precisions)


### Manage the learning rate schedules
def log_lr_schedule(num_chunks_train, updates=4, base=0.02):
    ls = np.logspace(0.0, np.round(np.log10(num_chunks_train)), num = updates)
    changepts = ls.astype(int)
    changepts[0] = 0
    learning_rates = [base * np.float(np.power(10,-1.0 * i)) for i in range(len(ls))]

    return OrderedDict(zip(changepts,learning_rates))

def lin_lr_schedule(num_chunks_train, base=0.02, cap=0.0002, updates=15):
    ls = np.linspace(0.0, updates * num_chunks_train, num = updates).astype(int)
    changepts = ls.astype(int)
    changepts[0] = 0
    learning_rates = [np.maximum(base * np.float(np.power(10,-1.0 * i)), cap) for i in range(len(ls))]
    
    return OrderedDict(zip(changepts,learning_rates))

def current_learning_rate(schedule, idx):
    for i in schedule.keys():
        if idx >= i:
            current_lr = schedule[i]

    return current_lr


### Utility functions


def softmax(x): 
    m = np.max(x, axis=1, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=1, keepdims=True)

def entropy(x):
    h = -x * np.log(x)
    h[np.invert(np.isfinite(h))] = 0
    return h.sum(1)


def conf_matrix(p, t, num_classes):
    if p.ndim == 1:
        p = one_hot(p, num_classes)
    if t.ndim == 1:
        t = one_hot(t, num_classes)
    return np.dot(p.T, t)


def accuracy_topn(y, t, n=5):
    if t.ndim == 2:
        t = np.argmax(t, axis=1)
    
    predictions = np.argsort(y, axis=1)[:, -n:]    
    
    accs = np.any(predictions == t[:, None], axis=1)

    return np.mean(accs)


def find_project_root():
    return os.path.expanduser("~/projects/SeqDemote/")


def count_params(param_list):
    ''' count the total number of parameters i
    in a list of of torch Tensors'''
    num_params = 0
    for tensor in param_list:
        size = tensor.size()
        cumulative_product = 1
        for dimension in size:
            cumulative_product *= dimension
        num_params += cumulative_product
    return num_params    