import torch  
import torch.nn as nn  
import torch.nn.functional as F 
import torchvision.models as models  
import numpy as np

eps = 1e-7

def abs_rel_error(pred, gt):
	#abs_rel_error = torch.mean(torch.abs(pred -gt/(gt + eps)))
	abs_rel_error = torch.abs((pred -gt)/(gt + eps))
	abs_rel_error[gt == 0] = 0
	num_pixels = torch.sum(gt != 0, dim=(1,2))
	abs_rel_error = torch.mean(torch.sum(abs_rel_error, dim=(1,2))/num_pixels)
	return abs_rel_error

def RMSE(pred, gt):
	#RMSE = torch.sqrt(torch.mean(torch.square(pred - gt)))
	RMSE = torch.square(pred - gt)
	RMSE[gt == 0] = 0
	num_pixels = torch.sum(gt != 0, dim=(1,2))
	RMSE = torch.mean(torch.sqrt(torch.sum(RMSE, dim=(1,2))/num_pixels))
	return RMSE

def sq_rel_error(pred, gt):
	#sq_rel_error = torch.mean(torch.square(pred -gt)/(gt + eps))
	sq_rel_error = torch.square(pred -gt)/(gt + eps)
	sq_rel_error[gt == 0] = 0
	num_pixels = torch.sum(gt != 0, dim=(1,2))
	sq_rel_error = torch.mean(torch.sum(sq_rel_error, dim=(1,2))/num_pixels)
	return sq_rel_error

def error_less_than(pred, gt, value=1.25):
	error = torch.abs(pred - gt)
	delta = torch.sum(torch.logical_and(error < value, gt != 0), dim=(1,2))/torch.sum(gt != 0, dim=(1,2))
	delta = torch.mean(delta)
	return delta

def norm_error(pred, gt, norm = 2):
	error = pred-gt
	error[gt == 0] = 0
	num_pixels = torch.sum(gt != 0, dim=(1,2))
	if norm == 2:
		error = torch.mean(torch.sqrt(torch.sum(torch.square(error), dim=(1,2)))/num_pixels)
	elif norm == 1:
		error = torch.mean(torch.sum(torch.abs(error), dim=(1,2))/num_pixels)
	else: 
		print('error norm not specified.')
		error = -1
	return error
