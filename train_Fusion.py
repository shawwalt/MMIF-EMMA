# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.getcwd())

import time
import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import warnings
import numpy as np
warnings.filterwarnings('ignore')  
import logging
logging.basicConfig(level=logging.CRITICAL)

from solver.solvers import EMMAExpSolver
import random

def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 2069
seed_everything(seed)

torch.set_default_dtype(torch.float32)

# 加载训练验证集
if __name__ == "__main__":
    solver = EMMAExpSolver("./MMIF-EMMA/option.yaml")
    solver.train()
    solver.test()
    solver.save_checkpoint()





