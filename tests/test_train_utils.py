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
        

def make_classification_labels_and_preds(shape=(128,164), p=0.1, flips=10):
    ''' fixture generator for mt_aupr / mt_auroc 
    returns labels, y_hat '''
    
    labels = np.random.binomial(1,p,size=shape)     
    preds = labels.copy()
    
    for data_pt in preds:
        data_pt = flip_random(data_pt, flips)
        
    return labels, preds
 

def test_st_accuracy():
    ''' make sure ST accuracy works '''
    test_labels, test_preds = make_classification_labels_and_preds()
    test_labels = test_labels[:,0]
    test_preds = test_preds[:,0]
    test_accuracy = tr_utils.st_accuracy(test_preds, test_labels)
    ok_(0.5 <= test_accuracy < 1.0)

def test_mt_accuracy():
    ''' make sure MT accuracy works '''
    test_labels, test_preds = make_classification_labels_and_preds()
    test_accuracy = tr_utils.mt_accuracy(test_preds, test_labels)
    ok_(0.5 <= test_accuracy < 1.0)
    
def test_mt_precision():
    ''' make sure MT precision works '''
    test_labels, test_preds = make_classification_labels_and_preds()
    test_precision = tr_utils.mt_precision(test_preds, test_labels)
    ok_(0.0 < test_precision < 1.0)    