import numpy as np 
import gzip
from sklearn.metrics import roc_auc_score, average_precision_score

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

def mt_accuracy(y, t):
    """ 
    multi-task ROC: the un-weighted average of task ROC scores
    """
    if not y.shape == t.shape:
        print("Error in AUC: shape mismatch for y: ", y.shape, " and t: ", t.shape)
        return -1
    rocs = []
    for preds, targets in zip(y.transpose(), t.transpose()):
        rocs.append(roc_auc_score(targets, preds))
    
    return np.mean(rocs)

def mt_precision(y, t):
    """
    multi-task precision: the un-weighted average of task precision scores
    """
    if not y.shape == t.shape:
        print("Error in precision: shape mismatch for y: ", y.shape, " and t: ", t.shape)
        return -1
    precisions = []
    for preds, targets in zip(y.transpose(), t.transpose()):
        precisions.append(average_precision_score(targets, preds))
    
    return np.mean(precisions)


### Manage the learning rate schedules

def current_learning_rate(schedule, idx):
    s = schedule.keys()
    s.sort()
    current_lr = schedule[0]
    for i in s:
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




