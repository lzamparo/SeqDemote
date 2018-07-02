import os 
import numpy as np
from nose.tools import eq_, ok_

import utils.train_utils as tr_utils


def flip_random(data, num_labels):
    ''' return a row of 0,1 labels with num_labels flipped '''
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

def make_classification_labels_and_preds(shape=(128,164), p=0.1, flips=10, noisy=False):
    ''' fixture generator for mt_aupr / mt_auroc 
    returns labels, y_hat '''
    
    labels = np.random.binomial(1,p,size=shape)     
    preds = np.array(labels.copy(), dtype=np.float)
    
    for data_pt in preds:
        data_pt = flip_random(data_pt, flips)
    
    if noisy:
        preds = make_fuzzy_predictions(preds)
        
    return labels, preds
 

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