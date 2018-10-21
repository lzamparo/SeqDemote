import argparse
import os
import torch
import importlib.util

import numpy as np
from torch.utils.data import DataLoader
from utils import train_utils

from tensorboardX import SummaryWriter
from torchviz import make_dot, make_dot_from_trace

parser = argparse.ArgumentParser()
parser.add_argument("config", help='path to config file')
parser.add_argument("--logdir", help='write tensorboard logs here', 
                    default='~/projects/SeqDemote/results/tb_logs')
args = parser.parse_args()

### Load up a model to visualize it with either tensorboardX or torchviz
model_config = args.config
model_path_name = os.path.join(os.path.expanduser(os.getcwd()),'models',model_config)
spec = importlib.util.spec_from_file_location(model_config, model_path_name)
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)

print("...Build model, load model state from save state if provided")
model = model_module.net

print("...Getting shape of dummy data", flush=True)
if hasattr(model_module, 'batch_size'):
    batch_size = model_module.batch_size
    embedding_dim = model_module.embedding_dim_len
    num_peaks = model_module.embedded_seq_len // embedding_dim
else:
    batch_size = 128

# generate dummy data, pass through the model
x = torch.randn((batch_size, 1, num_peaks, embedding_dim))
y = model(x)

# try to draw the model
# ... using Tensorbard logging
#writer = SummaryWriter(os.path.expanduser(args.logdir))
#writer.add_graph(model, x)

# ... using pytorchviz make_dot
g = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))
g.save(filename="model_fig.dot", directory=os.path.expanduser(args.logdir))

# ... using pytorchviz make_dot from trace 
with torch.onnx.set_training(model, False):
    trace, _ = torch.jit.get_trace_graph(model, args=(x,))
g = make_dot_from_trace(trace)
g.save(filename="model_fig_jit.dot", directory=os.path.expanduser(args.logdir))

#x = torch.randn(1, 3, 227, 227).requires_grad_(True)
#y = model(x)
#make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))