import numpy as np 
import os
from collections import OrderedDict
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef, precision_recall_curve


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
    #if params is None:
        #params = updates.keys()
    #updates = OrderedDict(updates)
    
    #if velocity_updates is None:
        #velocity_updates = OrderedDict()
        #for param in params:
            #value = param.get_value(borrow=True)
            #velocity_updates[param] = [theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                 #broadcastable=param.broadcastable) for i in range(momentums.shape[0])]

    ## calculate update expressions, using velocity_updates to update each velocity vector 
    ## prior to calculating the update for each model parameter
    #for param in params:
        
        ## calculate each velocity update
        #velocity_updates[param] = [momentum * velocity + updates[param] for momentum, velocity in zip(
            #velocity_updates[param], momentums)]
        
        ## calculate parameter update
        ##x = m * velocities + updates[param] 
        #K = momentums.shape[0]
        #updates[param] = updates[param] + (1. / K)  * sum(velocity_updates[param])

    
    #return updates, velocity_updates
    pass


def per_task_loss(y_hat, y, loss, do_sum=True):
    ''' Calculate the per-task loss.  The shape of y, y_hat 
    might be the same (tasks are all encoded in an NP array) or 
    y_hat will be a list of tasks, with length equal to y.shape[1]
    '''
    
    try:
        all_task_losses = [loss(y_hat[:,c], y[:,c]) for c in range(y_hat.shape[1])]
    except (AttributeError, AssertionError) as e:
        all_task_losses = [loss(y_hat[c].squeeze_(), y[:,c]) for c in range(y.shape[1])]
    if do_sum:
        return sum(all_task_losses)
    else:
        return all_task_losses
    

### Log-loss calculating utils

def one_hot(vec, m=None):
    if m is None:
        m = int(np.max(vec)) + 1

    return np.eye(m)[vec]


def thresholded(y_hat, thresh=0.5):
    """ Return thresholded (i.e predicted) values
    from a vector of scores """
    thresholded_y_hat = np.empty_like(y_hat)
    for i, pred in enumerate(y_hat):
        thresholded_y_hat[i] = 1.0 if pred > thresh else 0.0
    return thresholded_y_hat
        

def st_accuracy(y, y_hat):
    """ single-task (peak vs flank) ROC ;
    y_hat := predicted labels
    y := actual labels from data """
    if not y_hat.shape == y.shape:
        print("Error in single task AUC: shape mismatch in \hat{y}: ", y_hat.shape, " and y: ", y.shape)
        return -1
    return roc_auc_score(y, y_hat)

def validate_yhat(y, y_hat):
    """ ensure shapes match, and y_hat values \in [0,1]
    y_hat := predicted labels
    y := actual labels
    
    y_hat may be a list of arrays, or an array
    """
    if isinstance(y_hat, list):
        y_hat_array = np.vstack(y_hat).transpose()
        if y_hat_array.shape != y.shape:
            y_hat_array = np.hstack(y_hat).transpose()
        shape_is_valid = _shape_test(y, y_hat_array)
        values_are_valid = _value_test(y, y_hat_array)
    else:
        shape_is_valid = _shape_test(y, y_hat)
        values_are_valid = _value_test(y, y_hat)
    
    return shape_is_valid and values_are_valid

def _shape_test(y, y_hat):
    """ shape test for y, y_hat """
    
    shape_test = y_hat.shape == y.shape
    if not shape_test:
        print("Shape mismatch for \hat{y}: ", y_hat.shape, " and y: ", y.shape)
        return False
    
    return True

def _value_test(y, y_hat):
    """ value test for y, y_hat """
    
    zeros_test_array = y_hat >= 0
    ones_test_array = y_hat <= 1
    values_test_array = np.logical_or(zeros_test_array, ones_test_array)
    values_test = np.all(values_test_array)
    if not values_test:
        print("Values error for \hat{y}: ", y_hat[~values_test])
        return False
    
    return True

def mt_accuracy(y, y_hat, average=True):
    """ multi-task ROC: the un-weighted average of task ROC scores; 
    y_hat := predicted labels
    y := actual labels from data
    """
    if not validate_yhat(y, y_hat):
        return -1
    rocs = []
    for targets, preds in zip(y.transpose(), y_hat.transpose()):
        rocs.append(roc_auc_score(targets, preds))
    
    if average:
        return np.mean(rocs)
    else:
        return rocs

def mt_avg_precision(y, y_hat, average=True):
    """
    multi-task precision: the un-weighted average of task precision scores;
    y_hat := predicted labels
    y := actual labels from data
    """
    if not validate_yhat(y, y_hat):
        return -1
    precisions = []
    for targets, preds in zip(y.transpose(), y_hat.transpose()):
        precisions.append(average_precision_score(targets, preds))
    
    if average:
        return np.mean(precisions)
    else:
        return precisions
    
def mt_precision_at_recall(y, y_hat, average=True, recall_lvl=0.5):
    """
    multi-task precision at a given level of recall;
    y_hat := predicted labels
    y := actual labels
    
    Assume y_hat is same shape as y, otherwise it is a list.
    """
    
    if not validate_yhat(y, y_hat):
        return -1
    
    precisions_at_recall = []
    try:
        for targets, preds in zip(y.transpose(), y_hat.transpose()):
            precision, recall, thresholds = precision_recall_curve(targets, preds)
            idx = (np.abs(recall - recall_lvl)).argmin()  # index of element in recall array closest to recall_lvl
            precisions_at_recall.append(precision[idx])
    
    except AttributeError:
        for targets, preds in zip(y.transpose(), y_hat):
            precision, recall, thresholds = precision_recall_curve(targets, preds)
            idx = (np.abs(recall - recall_lvl)).argmin()  # index of element in recall array closest to recall_lvl
            precisions_at_recall.append(precision[idx])            
    
    if average:
        return np.mean(precisions_at_recall)
    else:
        return precisions_at_recall    

def mt_avg_f1_score(y, y_hat, average=True):
    """
    multi-task f1 score: un-weighted f1 scores;
    y_hat := predicted labels
    y := actual labels from data
    
    Assume y_hat is same shape as y, otherwise it is a list.
    """
    
    
    if not validate_yhat(y, y_hat):
        return -1
    
    f1_scores = []
    try:
        for targets, preds in zip(y.transpose(), y_hat.transpose()):
            f1_scores.append(f1_score(targets, thresholded(preds)))
    
    except AttributeError:
        for targets, preds in zip(y.transpose(), y_hat):
            f1_scores.append(f1_score(targets, thresholded(preds)))        
    
    if average:
        return np.mean(f1_scores)
    else:
        return f1_scores

def mt_avg_mcc(y, y_hat, average=True):
    ''' 
    multi-task MCC score: unweighted MCC scores
    y_hat := predicted labels
    y := actual labels
    
    Assume y_hat is same shape as y, otherwise it is a list.
    '''
    
    if not validate_yhat(y, y_hat):
        return -1
    
    mcc_scores = []
    try:
        for targets, preds in zip(y.transpose(), y_hat.transpose()):
            mcc_scores.append(matthews_corrcoef(targets, thresholded(preds)))
    except AttributeError:
        for targets, preds in zip(y.transpose(), y_hat):
            mcc_scores.append(matthews_corrcoef(targets, thresholded(preds)))
            
    if average:
        return np.mean(mcc_scores)
    else:
        return mcc_scores
    
    
### Manage the learning rate schedules
def log_lr_schedule(num_chunks_train, updates=4, base=0.02):
    ls = np.logspace(0.0, np.round(np.log10(num_chunks_train)), num = updates)
    changepts = ls.astype(int)
    changepts[0] = 0
    learning_rates = [base * np.float(np.power(10,-1.0 * i)) for i in range(len(ls))]

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