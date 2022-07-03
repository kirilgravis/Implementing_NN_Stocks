from dotenv import load_dotenv
import os
import pymongo
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

load_dotenv()
