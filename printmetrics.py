from preprocess import get_std
import cv2
import numpy as np
import torch.nn as nn
from data import getDataset, get_inverse_transforms
from torch.utils.data import DataLoader, Dataset
import torch
from modelv2 import Segnet
from modelv3 import Segnet as SegnetSkip3
#from predict import compute_accuracy
import os
from test_segmentation_camvid import label_colours
from model_dilated import SegmentationDilated as SegmentationDil
from model_dilated2 import SegmentationDilated as SegmentationDil2
from model_dilated3 import SegmentationDilated as SegmentationDil3
from model_dilated4 import SegmentationDilated as SegmentationDil4
from model_dilated5 import SegmentationDilated as SegmentationDil5
from model_dilated_attention import SegmentationDilated as SegmentationDilattn
from depthmodel import SwinTransformer, NeWCRFDepth
from data_config import rawdatalist, depthdatalist, valrawdatalist, valdepthdatalist
from depth_metrics import RMSE, abs_rel_error, sq_rel_error
from predict import compute_metrics, predict_and_visualize
from time import time
from torchsummary import summary
from losses import depth_loss
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import pickle

print('hi')

resultsdiropen = 'results/trial5'
with open(resultsdiropen + '/training_loss_list.pkl', 'rb') as f:
	training_loss_list = pickle.load(f)
	print('training_loss_list: ', training_loss_list)
        #with open(resultsdir + '/mean_list.pkl', 'rb') as f:
        #    mean_list = pickle.load(f)
with open(resultsdiropen + '/absrellist.pkl', 'rb') as f:
	absrellist = pickle.load(f)
	print('\nabsrellist: ', absrellist)
with open(resultsdiropen + '/rmselist.pkl', 'rb') as f:
	rmselist = pickle.load(f)
	print('\nrmselist: ', rmselist)
with open(resultsdiropen + '/absrel_val_list.pkl', 'rb') as f:
	absrel_val_list = pickle.load(f)
	print('\nabsrel_val_list: ', absrel_val_list)
with open(resultsdiropen + '/rmse_val_list.pkl', 'rb') as f:
	rmse_val_list = pickle.load(f)
	print('\nrmse_val_list: ', rmse_val_list)
with open(resultsdiropen + '/loss_val_list.pkl', 'rb') as f:
	loss_val_list = pickle.load(f)
	print('\nloss_val_list: ', loss_val_list)