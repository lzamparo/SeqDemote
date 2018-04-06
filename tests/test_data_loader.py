import load
from utils import dna_io

from nose.tools import eq_, ok_
from nose import SkipTest
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

### Basset DNase data fixtures
path = os.path.expanduser("~/projects/SeqDemote/data/DNase/encode_roadmap_all.h5")
chunk_size = 4096
batch_size = 128
batch_train_shape = (batch_size,4,1,600)

basset_train_size = 3201397
basset_valid_size = 70000
basset_output_size = 164
basset_num_chunks_train = basset_train_size // chunk_size
basset_num_chunks_valid = basset_valid_size // chunk_size
basset_full_train_shape = (basset_train_size, 4, 1, 600)
basset_chunk_train_shape = (chunk_size,4,1,600)
basset_chunk_out_shape = (chunk_size,basset_output_size)

### Hematopoetic DNase data fixtures
heme_path = os.path.expanduser("~/projects/SeqDemote/data/DNase/hematopoetic_data.h5")
heme_train_size = 240943
heme_valid_size = 51638
heme_output_size = 6

heme_num_chunks_train = heme_train_size // chunk_size
heme_num_chunks_valid = heme_valid_size // chunk_size

heme_full_train_shape = (heme_train_size, 4, 1, 600)
heme_chunk_train_shape = (chunk_size,4,1,600)
heme_chunk_out_shape = (chunk_size,heme_output_size)


### Kmer fixtures

aaa_3mer_string = 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'
aaa_3mer_position = 0

ttt_3mer_string = 'TTTTTTTTTTTTTTTTTTTTTTTTTTTTTT'
ttt_3mer_position = 63

aaa_4mer_string = 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'
ttt_4mer_string = 'TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT'

rando_3mer_string = 'ATGGGGTAGAGAATGGGGTAGAGACCAGGT'

alphabet_size_3 = 64
alphabet_size_4 = 256

threemer_chunk_shape = (chunk_size,alphabet_size_3, 1, 598)
fourmer_chunk_shape = (chunk_size, alphabet_size_4, 1, 597)


### Structure, shape based tests for Basset data

def test_build_data_loader():
    """ Can I build a data loader for the DNase data """
    data_loader = load.StandardDataLoader()
    data_loader.load_train()
    eq_(data_loader.train_in.shape, basset_full_train_shape)
    
def test_build_data_loader_kwargs():
    """ Can I build a data loader for the DNase data specifying the data path """
    
    data_loader = load.StandardDataLoader(data_path=path)
    data_loader.load_train()
    eq_(data_loader.train_in.shape, basset_full_train_shape)    

def test_dnase_data_shape():
    """ Is my DNase data the right size and shape """
    data_loader = load.StandardDataLoader(data_path=path)
    data_loader.load_train()
    num_chunks = range(basset_num_chunks_train)
    for e, (x_chunk, y_chunk) in zip(num_chunks,data_loader.create_batch_gen()):
        print("Chunk: ", str(e + 1), " of ", basset_num_chunks_train)
        eq_(x_chunk.shape, basset_chunk_train_shape)
        eq_(y_chunk.shape, basset_chunk_out_shape)
        
        
def test_exhaust_data():
    """ If I iterate through all the chunks, how many data points do I see? """
    seen_pts = 0
    data_loader = load.StandardDataLoader(data_path=path)
    data_loader.load_train()
    num_chunks = range(basset_num_chunks_train)
    for e, (x_chunk, y_chunk) in zip(num_chunks,data_loader.create_batch_gen()):
        seen_pts = seen_pts + x_chunk.shape[0]
    print("Saw ", str(seen_pts), " points total")    
    ok_(seen_pts <= basset_train_size) 
                
### Structure, shape based tests for Heme data        
        
def test_build_heme_data_loader():
    """ Can I build a data loader for the Hematopoetic data """
    data_loader = load.HematopoeticDataLoader()
    data_loader.load_train()
    eq_(data_loader.train_in.shape, heme_full_train_shape)
    
def test_build_heme_data_loader_kwargs():
    """ Can I build a data loader for the Heme data specifying the data path """
    
    data_loader = load.HematopoeticDataLoader(data_path=heme_path)
    data_loader.load_train()
    eq_(data_loader.train_in.shape, heme_full_train_shape)    
    
def test_build_heme_data_loader_pvsf():
    """ Can I build a peaks versus flanks output shape? """
    data_loader = load.HematopoeticDataLoader(data_path=heme_path, peaks_vs_flanks=True)
    data_loader.load_train()
    eq_(data_loader.train_out.shape, (heme_train_size,1))    
    

def test_heme_data_shape():
    """ Is my Hematopoetic data the right size and shape """
    data_loader = load.HematopoeticDataLoader(data_path=heme_path)
    data_loader.load_train()
    num_chunks = range(heme_num_chunks_train)
    for e, (x_chunk, y_chunk) in zip(num_chunks,data_loader.create_batch_gen()):
        print("Chunk: ", str(e + 1), " of ", heme_num_chunks_train)
        eq_(x_chunk.shape, heme_chunk_train_shape)
        eq_(y_chunk.shape, heme_chunk_out_shape)  
        
    

### Semantic tests: are all examples properly one-hot encoded?
    
def test_training_batch_encoding_sum():
    """ If I sum all elements of a chunk of training data, do I get the number of expected ones? """
    data_loader = load.StandardDataLoader(data_path=path)
    data_loader.load_train()
    num_chunks = range(basset_num_chunks_train)
    expected_chunk_sum = 600 * chunk_size
    ### each chunk of should have |seq_length| * |batch_size| * |chunk_size| number of 1s
    for e, (x_chunk, y_chunk) in zip(num_chunks,data_loader.create_batch_gen()):
        ones_per_chunk = np.sum(x_chunk)
        print("Chunk  ", str(e + 1), " has ", ones_per_chunk, " bits turned on")
        eq_(ones_per_chunk, expected_chunk_sum)
        
def test_validation_batch_encoding_sum():
    """ If I sum all elements of a chunk of validation data, do I get the number of expected ones? """
    data_loader = load.StandardDataLoader(data_path=path)
    data_loader.load_validation()
    num_chunks = range(basset_num_chunks_train)
    expected_chunk_sum = 600 * chunk_size
    ### each chunk of should have |seq_length| * |batch_size| * |chunk_size| number of 1s
    for e, (x_chunk, y_chunk) in zip(num_chunks,data_loader.create_valid_gen()):
        ones_per_chunk = np.sum(x_chunk)
        print("Chunk  ", str(e + 1), " has ", ones_per_chunk, " bits turned on")
        eq_(ones_per_chunk, expected_chunk_sum)        
        
        
def test_any_always_negatives_training():
    """ Are there any always-negative examples in the training set? """
    data_loader = load.StandardDataLoader(data_path=path)
    data_loader.load_train()    
    for e, (x_chunk, y_chunk) in zip(range(basset_num_chunks_train),data_loader.create_batch_gen()):
        for label_vector in y_chunk:
            total_peaks_on = np.sum(label_vector)
            ok_(total_peaks_on > 0)    
            
def test_any_always_negatives_validation():
    """ Are there any always-negative examples in the validation set? """
    data_loader = load.StandardDataLoader(data_path=path)
    data_loader.load_validation()    
    for e, (x_chunk, y_chunk) in zip(range(basset_num_chunks_valid),data_loader.create_valid_gen()):
        for label_vector in y_chunk:
            total_peaks_on = np.sum(label_vector)
            ok_(total_peaks_on > 0)
            

def test_enumerate_malformed_validation_data():
    """ How many mal-encoded validation data do we have? """
    
    raise SkipTest
    data_loader = load.StandardDataLoader(data_path=path)
    data_loader.load_validation()
    num_chunks = range(basset_num_chunks_train)
    expected_chunk_sum = 600 * chunk_size
    
    mistakes = []
    ### each chunk of should have |seq_length| * |chunk_size| number of 1s
    for e, (x_chunk, y_chunk) in zip(num_chunks,data_loader.create_valid_gen()):
        for item, element in enumerate(x_chunk):
            for i, position in enumerate(np.transpose(element)):
                ones_per_position = np.sum(position)
                if ones_per_position != 1.0:
                    mistakes.append("Error in position {0} of element {1} for chunk {2}".format(i, item, e))    
                #eq_(ones_per_position, 1.0)
    
    print("found ", len(mistakes), " mistakes total.")    
    for m in mistakes:
        print(m)
        
### Kmerizing generators / data-loader tests
        
def test_encoding_kmerizing_threemers():
    
    # encode the fixtures as a one-hot encoded matrix
    my_seqs = [aaa_3mer_string,ttt_3mer_string,aaa_3mer_string,ttt_3mer_string]    
    seqs = np.vstack([dna_io.dna_one_hot(seq) for seq in my_seqs])
    
    # reshape into expected shape
    seqs = seqs.reshape((seqs.shape[0],4,1,seqs.shape[1]/4))
    
    # re-encode the one-hot examples as positional 3-mers
    seqs_3mers = dna_io.one_hot_to_kmerized(seqs, 3)
    seqs_3mers_reshaped = seqs_3mers.reshape((seqs_3mers.shape[0],alphabet_size_3,1,len(aaa_3mer_string)-2))
    
    # make up the positional 3-mer fixture
    aaa_vec = dna_io.dna_one_hot_kmer(aaa_3mer_string, 3)
    aaa_mat = aaa_vec.reshape((1,alphabet_size_3,1,len(aaa_3mer_string) - 2))
    ttt_vec = dna_io.dna_one_hot_kmer(ttt_3mer_string, 3)
    ttt_mat = ttt_vec.reshape((1,alphabet_size_3,1,len(ttt_3mer_string) - 2))
    
    fixture = np.vstack((aaa_mat,ttt_mat,aaa_mat,ttt_mat))
    
    for fix, elem in zip(fixture,seqs_3mers_reshaped):
        ok_(np.allclose(fix, elem))
        
def test_decoding_kmerizing_threemers():
    
    # encode the fixtures as a one-hot encoded matrix
    my_seqs = [aaa_3mer_string,ttt_3mer_string,aaa_3mer_string,ttt_3mer_string]    
    seqs = np.vstack([dna_io.dna_one_hot(seq) for seq in my_seqs])
    
    # reshape into expected shape
    seqs = seqs.reshape((seqs.shape[0],4,1,seqs.shape[1]/4))
    
    # re-encode the one-hot examples as positional 3-mers
    seqs_3mers = dna_io.one_hot_to_kmerized(seqs, 3)
    seqs_3mers_reshaped = seqs_3mers.reshape((seqs_3mers.shape[0],alphabet_size_3,1,len(aaa_3mer_string)-2))  
    
    # decode the encoded 3-mers
    decoded_seqs = dna_io.kmer_vecs_to_dna(seqs_3mers_reshaped,3)
    for seq, fix in zip(decoded_seqs,my_seqs):
        eq_(seq,fix)
        
def test_encoding_kmerizing_fourmers():
    
    # encode the fixtures as one-hot encoded matrix
    my_seqs = [aaa_4mer_string,ttt_4mer_string, aaa_4mer_string, ttt_4mer_string]
    seqs = np.vstack([dna_io.dna_one_hot(seq) for seq in my_seqs])
    
    # reshape into expected shape
    seqs = seqs.reshape((seqs.shape[0], 4, 1, seqs.shape[1]/4))
    
    # re-encode the one-hot examles as positional 4-mers
    seqs_4mers = dna_io.one_hot_to_kmerized(seqs, 4)
    seqs_4mers_reshaped = seqs_4mers.reshape((seqs_4mers.shape[0],alphabet_size_4,1,len(aaa_4mer_string)-3))
    
    # make up the positional 4-mer fixture
    aaa_vec = dna_io.dna_one_hot_kmer(aaa_4mer_string,4)
    aaa_mat = aaa_vec.reshape((1,alphabet_size_4,1,len(aaa_4mer_string)-3))
    ttt_vec = dna_io.dna_one_hot_kmer(ttt_4mer_string, 4)
    ttt_mat = ttt_vec.reshape((1,alphabet_size_4,1,len(ttt_4mer_string)-3))
    
    fixture = np.vstack((aaa_mat,ttt_mat,aaa_mat,ttt_mat))
    
    for fix, elem in zip(fixture, seqs_4mers_reshaped):
        ok_(np.allclose(fix,elem))
    

def test_decoding_kmerizing_fourmers():
    
    # encode the fixtures as one-hot encoded matrix
    my_seqs = [aaa_4mer_string, ttt_4mer_string, aaa_4mer_string, ttt_4mer_string]
    seqs = np.vstack([dna_io.dna_one_hot(seq) for seq in my_seqs])
    
    # reshape into expected shape
    seqs = seqs.reshape((seqs.shape[0], 4, 1, seqs.shape[1]/4))
    
    # re-encode the one-hot examples as positional 4-mers
    seqs_4mers = dna_io.one_hot_to_kmerized(seqs,4)
    seqs_4mers_reshaped = seqs_4mers.reshape((seqs_4mers.shape[0], alphabet_size_4, 1, len(aaa_4mer_string)-3))
    
    # decode the encoded 4-mers
    decoded_seqs = dna_io.kmer_vecs_to_dna(seqs_4mers_reshaped, 4)
    for seq, fix in zip(decoded_seqs,my_seqs):
        eq_(seq,fix)
        

def test_threemer_data_loader_shape():
    """ Can I build a data loader for the kmerized DNase data specifying the data path? """
    data_loader = load.KmerDataLoader(data_path=path, kmer_length=3)
    data_loader.load_train()
    num_chunks = range(basset_num_chunks_train)
    for e, (x_chunk, y_chunk) in zip(num_chunks,data_loader.create_batch_gen()):
        print("Chunk: ", str(e + 1), " of ", basset_num_chunks_train)
        eq_(x_chunk.shape, threemer_chunk_shape)
        eq_(y_chunk.shape, basset_chunk_out_shape)

    
def test_fourmer_data_loader_shape():
    """ Can I build a data loader for the kmerized DNase data specifying the data path? """
    data_loader = load.KmerDataLoader(data_path=path, kmer_length=4)
    data_loader.load_train()
    num_chunks = range(basset_num_chunks_train)
    for e, (x_chunk, y_chunk) in zip(num_chunks,data_loader.create_batch_gen()):
        print("Chunk: ", str(e + 1), " of ", basset_num_chunks_train)
        eq_(x_chunk.shape, fourmer_chunk_shape)
        eq_(y_chunk.shape, basset_chunk_out_shape)    

def test_exhaust_threemer_data_loader():
    """ Is my kmerized data loader generator producing the right shape chunks? """
    seen_pts = 0
    data_loader = load.KmerDataLoader(data_path=path, kmer_length=3)
    data_loader.load_train()
    num_chunks = range(basset_num_chunks_train)
    for e, (x_chunk, y_chunk) in zip(num_chunks,data_loader.create_batch_gen()):
        seen_pts = seen_pts + x_chunk.shape[0]
    print("Saw ", str(seen_pts), " points total")    
    ok_(seen_pts <= basset_train_size)
    
    
### Kmer mismatch generator tests
def test_threemer_mismatch_range():
    # encode the fixtures as a mismatch encoded matrix
    my_seqs = [aaa_3mer_string,ttt_3mer_string,aaa_3mer_string,ttt_3mer_string]    
    seqs = np.vstack([dna_io.dna_mismatch_kmer(seq, kmer_length=3) for seq in my_seqs])
    
    # reshape into expected shape
    seqs = seqs.reshape((seqs.shape[0],4,1,seqs.shape[1]/4))
    
    # ensure all entries lie in [0,1]
    for seq in seqs:
        ok_(np.all(seq >= 0))
        ok_(np.all(seq <= 1))
        
def test_threemer_mismatch_all_range():
    data_loader = load.KmerDataLoader(data_path=path, kmer_length=4)
    data_loader.load_train()
    num_chunks = range(basset_num_chunks_train)
    for e, (x_chunk, y_chunk) in zip(num_chunks,data_loader.create_mismatch_batch_gen()):
        print("Chunk: ", str(e + 1), " of ", basset_num_chunks_train)
        for elem in x_chunk:
            ok_(np.all(elem >= 0))
            ok_(np.all(elem <= 1)) 
        
        