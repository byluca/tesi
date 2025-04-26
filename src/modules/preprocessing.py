"""
Definitions of data preprocessing pipelines.
"""
import os
import numpy as np
from monai.transforms import (
	CenterSpatialCropd,
	Compose,
	EnsureChannelFirstd,
	EnsureTyped,
	LoadImaged,
	MapTransform,
	NormalizeIntensityd,
	Orientationd,
	RandSpatialCropd,
	RandFlipd,
	RandScaleIntensityd,
	RandShiftIntensityd,
	Spacingd,
	ScaleIntensityRangePercentilesd
)
import torch


__all__ = ['get_transformations']


class LoadGraphCustomClassd(MapTransform):
	"""
	Loads graph data. It assumes that a graph dataset MUST BE generated.
	(Please see and run `src/graph_constructor.py` script before calling this class).
	"""
	def __init__(self, keys, graph_path, allow_missing_keys = False):
		super().__init__(keys, allow_missing_keys)
		self.graph_path = graph_path

	def __call__(self, data):
		d = dict(data)
		for key in self.keys:
			try:
				path = os.path.join(self.graph_path, d['subject'], d['subject'] + '.' + key)
				d[key] = torch.load(path)
			except OSError as e:
				print('\n' + ''.join(['> ' for i in range(30)]))
				print('\nERROR: file path for\033[95m '+path+'\033[0m not found.\n')
				print(''.join(['> ' for i in range(30)]) + '\n')
		return d


def get_transformations(graph_path = ''):
	"""
	Get data transformation pipelines.
	Args:
		graph_path (str): the path where to (optionally) load graphs.
	Returns:
		base_image_transform (monai.transforms.Compose): base pipeline for supervoxels extraction.
		autoencoder_train_transform (monai.transforms.Compose): pipeline for AE training input data.
		autoencoder_eval_transform (monai.transforms.Compose): pipeline for AE evaluation/testing input data.
	"""
	base_image_transform = Compose([
		LoadImaged(keys=['image', 'label']),
		EnsureChannelFirstd(keys=['image', 'label']),
		EnsureTyped(
			keys=['image', 'label'],
			data_type='numpy',
			dtype=np.float64
		),
		Orientationd(keys=['image', 'label'], axcodes='RAS'),
		Spacingd(
			keys=['image', 'label'],
			pixdim=(1.0, 1.0, 1.0),
			mode=('bilinear', 'bilinear'),
			align_corners=True,
			scale_extent=True
		),
		CenterSpatialCropd(keys=['image', 'label'], roi_size=(224, 224, 144)),
		ScaleIntensityRangePercentilesd(keys='image', lower=0, upper=99.5, b_min=0, b_max=1)
	])
	autoencoder_train_transform = Compose([
		LoadImaged(keys='image'),
		EnsureChannelFirstd(keys='image'),
		EnsureTyped(keys='image'),
		Orientationd(keys='image', axcodes='RAS'),
		Spacingd(
			keys='image',
			pixdim=(1.0, 1.0, 1.0),
			mode='bilinear',
			align_corners=True,
			scale_extent=True
		),
		RandSpatialCropd(keys='image', roi_size=(224, 224, 144)),
		RandFlipd(keys='image', prob=0.5, spatial_axis=0),
		RandFlipd(keys='image', prob=0.5, spatial_axis=1),
		RandFlipd(keys='image', prob=0.5, spatial_axis=2),
		NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
		RandScaleIntensityd(keys='image', factors=0.1, prob=1.0),
		RandShiftIntensityd(keys='image', offsets=0.1, prob=1.0)
	])
	autoencoder_eval_transform = Compose([
		LoadImaged(keys='image'),
		EnsureChannelFirstd(keys='image'),
		EnsureTyped(keys='image'),
		Orientationd(keys='image', axcodes='RAS'),
		Spacingd(
			keys='image',
			pixdim=(1.0, 1.0, 1.0),
			mode='bilinear',
			align_corners=True,
			scale_extent=True
		),
		CenterSpatialCropd(keys='image', roi_size=(224, 224, 144)),
		NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True)
	])
	gnn_train_eval_transform = Compose([
		LoadGraphCustomClassd(keys='graph', graph_path=graph_path),
		EnsureTyped(keys='graph')
	])
	gnn_test_transform = Compose([
		LoadGraphCustomClassd(keys=['graph', 'map'], graph_path=graph_path),
		LoadImaged(keys=['image', 'label']),
		EnsureChannelFirstd(keys=['image', 'label']),
		EnsureTyped(
			keys=['image', 'label'],
			data_type='numpy',
			dtype=np.float64
		),
		Orientationd(keys=['image', 'label'], axcodes='RAS'),
		Spacingd(
			keys=['image', 'label'],
			pixdim=(1.0, 1.0, 1.0),
			mode=('bilinear', 'bilinear'),
			align_corners=True,
			scale_extent=True
		),
		CenterSpatialCropd(keys=['image', 'label'], roi_size=(224, 224, 144))
	])
	return (
		base_image_transform,
		autoencoder_train_transform,
		autoencoder_eval_transform,
		gnn_train_eval_transform,
		gnn_test_transform
	)
