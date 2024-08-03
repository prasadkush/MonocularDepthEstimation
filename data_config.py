import numpy as np


rawdatapath = "D:/kitti raw images/2011_09_28_drive_0002_sync/2011_09_28/2011_09_28_drive_0002_sync/image_02/data"

depthdatapath = "D:/data_depth_annotated/train/2011_09_28_drive_0002_sync/proj_depth/groundtruth/image_02"

def get_mean_std(dataset_name='kitti'):
	if dataset_name == 'kitti':
		return mean, std