import os
import random

import numpy
import torch


class Reproducibility:
    @staticmethod
    def seed_all(seed=666):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
