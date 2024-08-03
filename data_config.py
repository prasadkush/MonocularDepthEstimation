import numpy as np


def get_mean_std(dataset_name='kitti'):
	if dataset_name == 'kitti':
		return mean, std