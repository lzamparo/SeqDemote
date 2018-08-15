import os 
import numpy as np
from nose.tools import eq_, ok_

import torch
import utils.train_utils as tr_utils
import utils.torch_model_construction_utils as tmu

def flip_random(data, num_labels):
    ''' return a column of 0,1 labels with num_labels flipped '''
    length = data.shape[0]
    flip_positions = np.random.randint(0,length,num_labels)
    for position in flip_positions:
        if data[position] == 0:
            data[position] = 1
        else:
            data[position] = 0
    return data
        

def make_fuzzy_predictions(preds, eps = 0.025, shape=0.1):
    ''' Add noise to 0-1 array to simulate predictions '''
    
    zeros = preds[preds == 0]
    ones  = preds[preds == 1]
    
    zero_noise = np.random.gamma(eps, shape, size=zeros.shape)
    ones_noise = -1.0 * np.random.gamma(eps, shape, size=ones.shape)
    
    noisy_zeros = zeros + zero_noise
    noisy_ones = ones + ones_noise
    
    preds[preds == 0] = noisy_zeros
    preds[preds == 1] = noisy_ones

    assert(np.alltrue(preds > 0))
    assert(np.alltrue(preds <= 1))
    return preds


def make_classification_labels_and_preds(shape=(128,164), p=0.1, 
                                         flips=10, noisy=False, 
                                         eps=0.025, g_shape=0.1):
    ''' fixture generator for mt_aupr / mt_auroc 
    returns labels, y_hat '''
    
    labels = np.random.binomial(1,p,size=shape)     
    preds = np.array(labels.copy(), dtype=np.float)
    
    for col in preds.transpose():
        col = flip_random(col, flips)
    
    if noisy:
        preds = make_fuzzy_predictions(preds, eps, g_shape)
    
    return labels, preds


def make_presigmoid_activations(preds, confident=True, to_tensor=False):
    ''' fixture generator for pre-sigmoid activations from 
    network output. Makes more confident predictions or less
    confident predictions'''
    
    extended_activations = np.zeros_like(preds)
    if confident:
        noise = np.random.gamma(5, 1, size=extended_activations.shape)
    else:
        noise = np.random.gamma(1,0.5, size=extended_activations.shape)
    
    # want to iterate elementwise here, maybe flatten / it / reshape?
    for e, p, n in zip(np.nditer(extended_activations, op_flags=[['readwrite']]), preds.flat, noise.flat):
        if p > 0.5:
            e += n
        else:
            e -= n
    
    if to_tensor:
        return torch.tensor(extended_activations)
    
    return extended_activations      

def test_focal_loss():
    ''' make sure focal loss increases for uncertain predictions '''
    
    ### If I need to compare the weights, losses pre-activations for each fixture 
    ### across all tasks, set reduce=False
    
    labels, preds = make_classification_labels_and_preds(shape=(4,4), flips=1)
    focal_loss = tmu.FocalLoss(reduce=True)    
    
    # generate certain predictions, calculate focal loss
    certain_activations = make_presigmoid_activations(preds, confident=True, to_tensor=True)
    certain_loss = tr_utils.per_task_loss(certain_activations, torch.tensor(labels, dtype=torch.double), focal_loss, do_sum=False)
   
    # generate less-certain predictions, calculate focal loss
    uncertain_activations = make_presigmoid_activations(preds, confident=False, to_tensor=True)    
    uncertain_loss = tr_utils.per_task_loss(uncertain_activations, torch.tensor(labels, dtype=torch.double), focal_loss, do_sum=False)
    
    # Q: should less-certain losses have much greater loss?
    # A: depends on the level of certainty (i.e p_t) and gamma.
    ok_(sum(uncertain_loss) < sum(certain_loss))

def test_st_accuracy():
    ''' make sure ST accuracy works '''
    test_labels, test_preds = make_classification_labels_and_preds()
    test_labels = test_labels[:,0]
    test_preds = test_preds[:,0]
    test_accuracy = tr_utils.st_accuracy(test_labels, test_preds)
    ok_(0.5 <= test_accuracy < 1.0)

def test_mt_accuracy():
    ''' make sure MT accuracy works '''
    test_labels, test_preds = make_classification_labels_and_preds()
    test_accuracy = tr_utils.mt_accuracy(test_labels, test_preds)
    ok_(0.5 <= test_accuracy < 1.0)
    
def test_mt_precision():
    ''' make sure MT precision works '''
    test_labels, test_preds = make_classification_labels_and_preds()
    test_precision = tr_utils.mt_avg_precision(test_labels, test_preds)
    ok_(0.0 < test_precision < 1.0)    
    
def test_noisy_mt_precision():
    ''' make sure MT precision works '''
    test_labels, test_preds = make_classification_labels_and_preds(noisy=True)
    test_precision = tr_utils.mt_avg_precision(test_labels, test_preds)
    ok_(0.0 < test_precision < 1.0)   
    
def test_mt_f1():
    ''' make sure MT f1 works '''
    test_labels, test_preds = make_classification_labels_and_preds(noisy=True)
    test_f1 = tr_utils.mt_avg_f1_score(test_labels, test_preds)
    ok_(0.0 < test_f1 < 1.0)

def test_mt_mcc():
    ''' make sure MT MCC works '''
    test_labels, test_preds = make_classification_labels_and_preds(noisy=True)
    test_mcc = tr_utils.mt_avg_mcc(test_labels, test_preds)
    ok_(-1.0 < test_mcc < 1.0)