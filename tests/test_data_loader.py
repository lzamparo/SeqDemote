import load


from nose.tools import eq_, ok_
import nose, functools
import os
import numpy as np


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
train_size = 1880000
valid_size = 70000
output_size = 164
chunk_size = 4096
batch_size = 128
num_chunks_train = train_size // chunk_size

full_train_shape = (train_size, 4, 1, 600)
chunk_train_shape = (chunk_size,4,1,600)
chunk_out_shape = (chunk_size,output_size)
batch_train_shape = (batch_size,4,1,600)

def test_build_data_loader():
    """ Can I build a data loader for the DNase data """
    data_loader = load.DNaseDataLoader()
    data_loader.load_train()
    eq_(data_loader.train_in.shape, full_train_shape)
    
def test_build_data_loader_kwargs():
    """ Can I build a data loader for the DNase data specifying the data path """
    path = os.path.expanduser("~/projects/SeqDemote/data/DNase/encode_roadmap.h5")
    data_loader = load.DNaseDataLoader(data_path=path)
    data_loader.load_train()
    eq_(data_loader.train_in.shape, full_train_shape)    

def test_dnase_data_shape():
    """ Is my DNase data the right size and shape """
    path = os.path.expanduser("~/projects/SeqDemote/data/DNase/encode_roadmap.h5")
    data_loader = load.DNaseDataLoader(data_path=path)
    data_loader.load_train()
    num_chunks = range(num_chunks_train)
    for e, (x_chunk, y_chunk) in zip(num_chunks,data_loader.create_batch_gen()):
        print("Chunk: ", str(e + 1), " of ", num_chunks_train)
        eq_(x_chunk.shape, chunk_train_shape)
        eq_(y_chunk.shape, chunk_out_shape)
    
def test_exhaust_data():
    """ If I iterate through all the chunks, how many data points do I see? """
    seen_pts = 0
    path = os.path.expanduser("~/projects/SeqDemote/data/DNase/encode_roadmap.h5")
    data_loader = load.DNaseDataLoader(data_path=path)
    data_loader.load_train()
    num_chunks = range(num_chunks_train)
    for e, (x_chunk, y_chunk) in zip(num_chunks,data_loader.create_batch_gen()):
        seen_pts = seen_pts + x_chunk.shape[0]
    print("Saw ", str(seen_pts), " points total")    
    ok_(seen_pts <= train_size)
