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
warnings.filterwarnings('ignore')  
import logging
logging.basicConfig(level=logging.CRITICAL)

from solver.solvers import EMMAExpSolver
import random

random.seed(2069)

torch.set_default_dtype(torch.float32)

# 加载训练验证集
if __name__ == "__main__":
    solver = EMMAExpSolver("./MMIF-EMMA/option.yaml")
    solver.train()
    solver.test()
    solver.save_checkpoint()





