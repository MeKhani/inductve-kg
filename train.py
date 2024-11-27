import os 
from tqdm import tqdm
import random
# from model import InGram
import numpy as np
import torch
from my_parser import parse
from dataset import TrainData



OMP_NUM_THREADS = 8
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True
torch.set_num_threads(8)
torch.cuda.empty_cache()

args = parse()
print(type(args))
print("run it ")
assert args.data_name in os.listdir(args.data_path), f"{args.data_name} Not Found"
path = args.data_path + args.data_name + "/"
train = TrainData(path)

