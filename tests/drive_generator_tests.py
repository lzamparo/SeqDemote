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

### DNase data fixtures
train_size = 3201397
valid_size = 70000
output_size = 164
chunk_size = 4096
batch_size = 128
num_chunks_train = train_size // chunk_size
num_chunks_valid = valid_size // chunk_size

full_train_shape = (train_size, 4, 1, 600)
chunk_train_shape = (chunk_size,4,1,600)
chunk_out_shape = (chunk_size,output_size)
batch_train_shape = (batch_size,4,1,600)

path = os.path.expanduser("~/projects/SeqDemote/data/DNase/encode_roadmap.h5")

### Kmer fixtures

alphabet_size_1 = 4
alphabet_size_3 = 64
alphabet_size_4 = 256
alphabet_size_5 = 1024

threemer_chunk_shape = (chunk_size,alphabet_size_3, 1, 598)
fourmer_chunk_shape = (chunk_size, alphabet_size_4, 1, 597)
fivemer_chunk_shape = (chunk_size, alphabet_size_5, 1, 596)

### 

h5file = h5py.File(path)
train_in = h5file['/train_in']
train_out = h5file['/train_out']
training_data = np.zeros(train_in.shape, dtype=train_in.dtype)
training_data[:] = train_in[:]
training_targets = np.zeros(train_out.shape,dtype=train_out.dtype)
h5file.close()

def get_training_data():
    return training_data, training_targets

### Structure, shape based tests

def test_create_batch_gen():
    """ Can I build a batch generator? """
    training_data, training_targets = get_training_data()
    my_gen = generators.train_sequence_gen(training_data, training_targets)
    for e, (x_chunk, y_chunk) in zip(range(num_chunks_train), my_gen):
        ok_(x_chunk.shape == chunk_train_shape)


def test_exhaust_data():
    """ If I iterate through all the chunks, how many data points do I see? """
    
    training_data, training_targets = get_training_data()
    seen_pts = 0
    num_chunks = range(num_chunks_train)
    my_gen = generators.train_sequence_gen(training_data, training_targets)
    for e, (x_chunk, y_chunk) in zip(num_chunks,my_gen):
        seen_pts = seen_pts + x_chunk.shape[0]
    print("Saw ", str(seen_pts), " points total")    
    ok_(seen_pts <= train_size)


def test_buffered_batch_gen_threaded():
    """ Can I create a buffered batch generator and exhaust the data? """
    
    training_data, training_targets = get_training_data()
    my_gen = buffering.buffered_gen_threaded(generators.train_sequence_gen(training_data, training_targets), buffer_size=3)
    num_chunks = range(num_chunks_train)
    seen_pts = 0
    for e, (x_chunk, y_chunk) in zip(num_chunks,my_gen):
        seen_pts = seen_pts + x_chunk.shape[0]
    print("Saw ", str(seen_pts), " points total")    
    ok_(seen_pts == train_size)
    

 
def test_buffered_batch_gen_mp():
    """ Can I create a buffered batch generator and exhaust the data? """
    
    training_data, training_targets = get_training_data()
    my_gen = buffering.buffered_gen_mp(generators.train_sequence_gen(training_data, training_targets), buffer_size=3)
    num_chunks = range(num_chunks_train)
    seen_pts = 0
    for e, (x_chunk, y_chunk) in zip(num_chunks,my_gen):
        seen_pts = seen_pts + x_chunk.shape[0]
    print("Saw ", str(seen_pts), " points total")    
    ok_(seen_pts == train_size)
    

### Kmerizing generators 


def test_buffered_kmerizing_gen():
    """ Can I create a buffered batch generator that kmerizes the data? """
    
    training_data, training_targets = get_training_data()
    my_gen = buffering.buffered_gen_mp(generators.train_kmerize_gen(training_data, training_targets, kmersize=3), buffer_size=3)
    num_chunks = range(num_chunks_train)
    seen_pts = 0
    for e, (x_chunk, y_chunk) in zip(num_chunks,my_gen):
        seen_pts = seen_pts + x_chunk.shape[0]
    print("Saw ", str(seen_pts), " points total")    
    ok_(seen_pts == train_size)




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


if __name__ == "__main__":
    #test_buffered_batch_gen_threaded()
    test_buffered_batch_gen_woozle()
    #test_buffered_batch_gen_mp()
    #test_buffered_kmerizing_gen()