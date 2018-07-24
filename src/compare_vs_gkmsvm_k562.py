import os
import h5py
import numpy as np
import subprocess
import tempfile
from pathlib import Path


def get_TF_labels_from_h5(h5handle, h5_path, TF_list):
    ''' get all labels from h5_path under the open handle,
    return only those column TFs that have a number of peaks 
    which overlap the K562 peak atlas'''
    
    TF_overlaps = [s.encode('utf-8') for s in TF_list]
    TF_colnames = h5handle[h5_path].attrs['column_names']
    col_mask_array = np.array([n in TF_overlaps for n in TF_colnames])
    all_labels = h5handle[h5_path][:]
    labels = all_labels[:,col_mask_array] 
    return labels


def generate_fasta(file):
    line_id = ""
    buffer = ""
    for line in file:
        line = line.strip()
        if line.startswith(">"):
            if len(buffer) > 0:
                yield line_id, buffer
            line_id = line.lstrip(">").rstrip()
            buffer = ""
        else:
            buffer = buffer + line

def process_fasta(seq_id, seq, outfile):
    print(seq_id, file=outfile)
    print(seq, file=outfile)    

def encode_sequences_into_fasta(seq_file, pos_t, neg_t, pos_v, neg_v,
                                mask_array, t_labels, v_labels):
    ''' Split the FASTA sequences into training, validation based on the training
    data / validation data binary mask array, and then further into 
    positive / negative files based on the corresponding label file'''
    
    train_index = 0
    valid_index = 0
    with open(seq_file,'r') as my_seqs:
        for (i,seq), is_valid in zip(generate_fasta(my_seqs), mask_array):
            my_id = ">{}".format(i)
            if is_valid:
                outfile = pos_v if v_labels[valid_index] else neg_v
                process_fasta(my_id, seq, outfile)
                valid_index += 1                
            else:
                outfile = pos_t if t_labels[train_index] else neg_t
                process_fasta(my_id, seq, outfile)
                train_index += 1                
        
            
def run_gkmtrain(positive_file, negative_file, model_prefix, gkm_prefix="~/projects/fixes/lsgkm/bin"):
    ''' call out to gkmtrain '''
    
    threads = 4
    word_len = 8
    kolumns = 5
    d_mismatches = 2
    mem_cache = 1000.0
    gkm_prefix_path = os.path.expanduser(gkm_prefix)
    gkm_train_cmd = os.path.join(gkm_prefix_path, "gkmtrain")
    command = [str(s) for s in [gkm_train_cmd, 
                                '-T', threads, 
                                '-l', word_len, 
                                '-k', kolumns,
                                '-d', d_mismatches, 
                                '-m', mem_cache, 
                                positive_file, 
                                negative_file, 
                                model_prefix]]
    subprocess.check_call(command)
    
def run_gkmpredict(input_file, output_file, model_file, gkm_prefix="~/projects/fixes/lsgkm/bin"):
    ''' call out to gkmpredict. '''
    
    threads = '-T 4'
    gkm_prefix_path = os.path.expanduser(gkm_prefix)
    gkm_predict_cmd = os.path.join(gkm_prefix_path, "gkmpredict")
    command = [str(s) for s in [gkm_predict_cmd,
                                threads,
                                input_file,
                                model_file,
                                output_file]]
    subprocess.run(command)
    
def gather_gkm_predictions(output_file):
    with open(output_file,'r') as f:
        lines = f.readlines()
    y_hat = np.array([line.split()[-1] for line in lines], dtype=float)
    return y_hat

TF_list = ["CEBPB","CEBPG", "CREB3L1", "CTCF",
           "CUX1","ELK1","ETV1","FOXJ2","KLF13",
           "KLF16","MAFK","MAX","MGA","NR2C2",
           "NR2F1","NR2F6","NRF1","PKNOX1","ZNF143"]
data_h5_path = os.path.expanduser("~/projects/SeqDemote/data/ATAC/K562/K562_embed_TV_annotated_split.h5")
fasta_path = os.path.expanduser("~/projects/SeqDemote/data/ATAC/K562/K562_atac.fasta")

# get the training, validation peak indices
with h5py.File(data_h5_path,'r') as h5:
    peak_index = h5["/data/peak_index"][:]
    training_labels = get_TF_labels_from_h5(h5, "/labels/training/train_labels", TF_list)
    validation_labels = get_TF_labels_from_h5(h5, "/labels/validation/valid_labels", TF_list)

validation_gkm_factor_dict = {}
results_prefix = os.path.expanduser("~/projects/SeqDemote/results/gkmsvm_comparison")
for pos,f in enumerate(TF_list):
    
    # for now: easiest is to train a per-factor model which is one vs all non-shared positives
    # extract the sequences for each factor
    pos_train = tempfile.NamedTemporaryFile(mode='w')
    neg_train = tempfile.NamedTemporaryFile(mode='w')
    train_peak_fname = pos_train.name
    train_flank_fname = neg_train.name
    
    pos_valid = tempfile.NamedTemporaryFile(mode='w')
    neg_valid = tempfile.NamedTemporaryFile(mode='w')
    valid_peak_fname = pos_valid.name
    valid_flank_fname = neg_valid.name
    
    encode_sequences_into_fasta(fasta_path, pos_train, 
                                neg_train, pos_valid, 
                                neg_valid, peak_index,
                                training_labels[:,pos],
                                validation_labels[:,pos])
    
    # train the model
    model_name = os.path.join(results_prefix, f)
    model_path = Path(model_name + ".model.txt")
    if not model_path.exists():
        # run gkmtrain
        run_gkmtrain(train_peak_fname, train_flank_fname, model_name)
    
    # validate the model
    valid_peaks_predictions_path = os.path.join(results_prefix,f + "_predicted_peaks.txt")
    valid_flanks_predictions_path = os.path.join(results_prefix,f + "_predicted_flanks.txt")
    run_gkmpredict(valid_peak_fname, valid_peaks_predictions_path, model_name + ".model.txt")
    valid_peak_preds = gather_gkm_predictions(valid_peaks_predictions_path)
    run_gkmpredict(valid_flank_fname, valid_flanks_predictions_path, model_name + ".model.txt")
    valid_flank_preds = gather_gkm_predictions(valid_flanks_predictions_path)
    
    pos_train.close()
    neg_train.close()    
    pos_valid.close()
    neg_valid.close()