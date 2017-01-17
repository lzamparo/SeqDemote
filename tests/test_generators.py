import generators
import buffering
import h5py

from nose.tools import eq_, ok_
from nose import SkipTest

import nose, functools
import os
import numpy as np

def with_setup_args(setup, teardown=None):
    """Decorator to add setup and/or teardown methods to a test function::

      @with_setup_args(setup, teardown)
      def test_something():
          " ... "

    The setup function should return (args, kwargs) which will be passed to
    test function, and teardown function.

    Note that `with_setup_args` is useful *only* for test functions, not for test
    methods or inside of TestCase subclasses.
    """
    def decorate(func):
        args = []
        kwargs = {}

        def test_wrapped():
            func(*args, **kwargs)

        test_wrapped.__name__ = func.__name__

        def setup_wrapped():
            a, k = setup()
            args.extend(a)
            kwargs.update(k)
            if hasattr(func, 'setup'):
                func.setup()
        test_wrapped.setup = setup_wrapped

        if teardown:
            def teardown_wrapped():
                if hasattr(func, 'teardown'):
                    func.teardown()
                teardown(*args, **kwargs)

            test_wrapped.teardown = teardown_wrapped
        else:
            if hasattr(func, 'teardown'):
                test_wrapped.teardown = func.teardown()
        return test_wrapped
    return decorate

def expected_failure(test):
    @functools.wraps(test)
    def inner(*args, **kwargs):
        try:
            test(*args, **kwargs)
        except Exception:
            raise nose.SkipTest
        else:
            raise AssertionError('Failure expected')
    return inner


### Common shape fixture params

chunk_size = 4096
batch_size = 128
batch_train_shape = (batch_size,4,1,600)
chunk_train_shape = (chunk_size,4,1,600)

### Basset DNase data fixtures
basset_train_size = 3201397  
basset_valid_size = 70000
basset_output_size = 164

basset_num_chunks_train = basset_train_size // chunk_size
basset_num_chunks_valid = basset_valid_size // chunk_size
basset_full_train_shape = (basset_train_size, 4, 1, 600)
basset_chunk_out_shape = (chunk_size,basset_output_size)

basset_path = os.path.expanduser("~/projects/SeqDemote/data/DNase/encode_roadmap_all.h5")

### Heme DNase data fixtures
heme_train_size = 240943
heme_valid_size = 51638
heme_output_size = 6

heme_num_chunks_train = heme_train_size // chunk_size
heme_num_chunks_valid = heme_valid_size // chunk_size
heme_full_train_shape = (heme_train_size, 4, 1, 600)
heme_chunk_out_shape = (chunk_size, heme_output_size)

heme_path = os.path.expanduser("~/projects/SeqDemote/data/DNase/hematopoetic_data.h5")

### Kmer fixtures

alphabet_size_1 = 4
alphabet_size_3 = 64
alphabet_size_4 = 256
alphabet_size_5 = 1024

threemer_chunk_shape = (chunk_size,alphabet_size_3, 1, 598)
fourmer_chunk_shape = (chunk_size, alphabet_size_4, 1, 597)
fivemer_chunk_shape = (chunk_size, alphabet_size_5, 1, 596)

### 


def get_basset_training_data(basset_path):
    h5file = h5py.File(basset_path)
    train_in = h5file['/train_in']
    train_out = h5file['/train_out']
    training_data = np.zeros(train_in.shape, dtype=train_in.dtype)
    training_data[:] = train_in[:]
    training_targets = np.zeros(train_out.shape,dtype=train_out.dtype)
    h5file.close()
    
    return training_data, training_targets

def get_heme_training_data(heme_path,peaks_vs_flanks=True):
    h5file = h5py.File(heme_path)
    peaks_train_in = h5file['/peaks/data/train_in']
    flanks_train_in = h5file['/flanks/data/train_in']
    n_peaks = peaks_train_in.shape[0]
    n_flanks = flanks_train_in.shape[0]
    train_set_size = n_peaks + n_flanks
    train_set_shape = tuple([n_peaks + n_flanks, peaks_train_in.shape[1], peaks_train_in.shape[2], peaks_train_in.shape[3]])
    train_in = np.zeros(train_set_shape, dtype=peaks_train_in.dtype)
    train_in[0:n_peaks,:,:,:] = peaks_train_in[:]
    train_in[n_peaks:n_peaks+n_flanks,:,:,:] = flanks_train_in[:]

    peaks_train_out = h5file['/peaks/labels/train_out']
    flanks_train_out = h5file['/flanks/labels/train_out']
    if peaks_vs_flanks:
        train_out = np.zeros(tuple([n_peaks + n_flanks,1]), dtype=peaks_train_out.dtype)
        train_out[0:n_peaks] = 1
    else:
        train_out = np.zeros(tuple([n_peaks + n_flanks,peaks_train_out.shape[1]]), dtype=peaks_train_out.dtype)
        train_out[0:n_peaks,:] = peaks_train_out[:]
        train_out[n_peaks:n_peaks+n_flanks,:] = flanks_train_out[:]

    h5file.close() 
    return train_in, train_out

### Structure, shape based tests

def test_create_basset_batch_gen():
    """ Can I build a batch generator? """
    training_data, training_targets = get_basset_training_data(basset_path)
    my_gen = generators.train_sequence_gen(training_data, training_targets)
    for e, (x_chunk, y_chunk) in zip(range(basset_num_chunks_train), my_gen):
        ok_(x_chunk.shape == chunk_train_shape)
        

def test_create_heme_batch_gen():
    """ Can I build a hematopoetic batch generator? """
    training_data, training_targets = get_heme_training_data(heme_path)
    my_gen = generators.train_sequence_gen(training_data, training_targets)
    for e, (x_chunk, y_chunk) in zip(range(heme_num_chunks_train), my_gen):
        ok_(x_chunk.shape == chunk_train_shape)

def test_create_buffered_heme_batch_gen():
    """ Can i build a buffered batch generator from the hematopoetic data? """
    training_data, training_targets = get_heme_training_data(heme_path)
    my_buffered_gen = buffering.buffered_gen_threaded(generators.train_sequence_gen(training_data, training_targets))
    num_chunks = range(heme_num_chunks_train)
    seen_pts = 0
    for e, (x_chunk, y_chunk) in zip(num_chunks, my_buffered_gen):
        seen_pts += x_chunk.shape[0]
        ok_(y_chunk.shape == (chunk_size,1))
    print("saw ", seen_pts, " points, expecting ", heme_train_size, " points")
    ok_((seen_pts == heme_train_size) or (abs(seen_pts - heme_train_size) < chunk_size))
    
        
def test_peaks_vs_flanks_gen():
    """ Does the Hematopoetic data generator return peaks vs flanks output? """
    training_data, training_targets = get_heme_training_data(heme_path)
    my_gen = generators.train_sequence_gen(training_data, training_targets)
    for e, (x_chunk, y_chunk) in zip(range(heme_num_chunks_train), my_gen):
        ok_(y_chunk.shape == (chunk_size,1))        

def test_exhaust_data():
    """ If I iterate through all the chunks, how many data points do I see? """
    
    training_data, training_targets = get_basset_training_data()
    seen_pts = 0
    num_chunks = range(basset_num_chunks_train)
    my_gen = generators.train_sequence_gen(training_data, training_targets)
    for e, (x_chunk, y_chunk) in zip(num_chunks,my_gen):
        seen_pts = seen_pts + x_chunk.shape[0]
    print("Saw ", str(seen_pts), " points total")    
    ok_(seen_pts <= basset_train_size)


def test_buffered_batch_gen_threaded():
    """ Can I create a buffered batch generator and exhaust the data? """
    
    training_data, training_targets = get_basset_training_data(basset_path)
    my_gen = buffering.buffered_gen_threaded(generators.train_sequence_gen(training_data, training_targets), buffer_size=3)
    num_chunks = range(basset_num_chunks_train)
    seen_pts = 0
    for e, (x_chunk, y_chunk) in zip(num_chunks,my_gen):
        seen_pts = seen_pts + x_chunk.shape[0]
    print("Saw ", str(seen_pts), " points total")    
    ok_((seen_pts == basset_train_size) or (abs(seen_pts - basset_train_size) < chunk_size))
    
 
def test_buffered_batch_gen_mp():
    """ Can I create a buffered batch generator and exhaust the data? """
    
    training_data, training_targets = get_basset_training_data(basset_path)
    my_gen = buffering.buffered_gen_mp(generators.train_sequence_gen(training_data, training_targets), buffer_size=3)
    num_chunks = range(basset_num_chunks_train)
    seen_pts = 0
    for e, (x_chunk, y_chunk) in zip(num_chunks,my_gen):
        seen_pts = seen_pts + x_chunk.shape[0]
    print("Saw ", str(seen_pts), " points total")    
    ok_(seen_pts == basset_train_size)
    

### Kmerizing generators 


def test_buffered_kmerizing_gen():
    """ Can I create a buffered batch generator that kmerizes the data? """
    
    training_data, training_targets = get_basset_training_data(basset_path)
    my_gen = buffering.buffered_gen_mp(generators.train_kmerize_gen(training_data, training_targets, kmersize=3), buffer_size=3)
    num_chunks = range(basset_num_chunks_train)
    seen_pts = 0
    for e, (x_chunk, y_chunk) in zip(num_chunks,my_gen):
        seen_pts = seen_pts + x_chunk.shape[0]
    print("Saw ", str(seen_pts), " points total")    
    ok_(seen_pts == basset_train_size)




#### Kmer mismatch generator tests
#def test_threemer_mismatch_range():
    ## encode the fixtures as a mismatch encoded matrix
    #my_seqs = [aaa_3mer_string,ttt_3mer_string,aaa_3mer_string,ttt_3mer_string]    
    #seqs = np.vstack([dna_io.dna_mismatch_kmer(seq, kmer_length=3) for seq in my_seqs])

    ## reshape into expected shape
    #seqs = seqs.reshape((seqs.shape[0],4,1,seqs.shape[1]/4))

    ## ensure all entries lie in [0,1]
    #for seq in seqs:
        #ok_(np.all(seq >= 0))
        #ok_(np.all(seq <= 1))

#def test_threemer_mismatch_all_range():
    #data_loader = load.KmerDataLoader(data_path=path, kmer_length=4)
    #data_loader.load_train()
    #num_chunks = range(num_chunks_train)
    #for e, (x_chunk, y_chunk) in zip(num_chunks,data_loader.create_mismatch_batch_gen()):
        #print("Chunk: ", str(e + 1), " of ", num_chunks_train)
        #for elem in x_chunk:
            #ok_(np.all(elem >= 0))
            #ok_(np.all(elem <= 1)) 

