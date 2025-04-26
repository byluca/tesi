"""
A set of functions for graph construction and manipulation
"""
import os
import numpy as np
from skimage.segmentation import slic
from skimage.measure import regionprops
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from src.helpers.utils import get_device, get_config
from src.modules.training import predict_ae
from src.modules.preprocessing import get_transformations


__all__ = ['image_to_graph']


def image_to_graph(
		data,
		n_segments = 15000,
		compactness = .1,
		sigma = 1,
		model = None,
		saved_path = None,
		percentiles = [10, 25, 50, 75, 90],
		k = 10,
		write_to_file = None
	):
	"""
	Converts a Dataloader entry to graph structure.
	Args:
		data (dict): an example returned from Dataloader class.

		n_segments (int): SLIC param, the (approximate) number of labels in the segmented output image.
		compactness (float): SLIC param, balances color proximity and space proximity. Higher values give
			more weight to space proximity, making superpixel shapes more square/cubic.
		sigma (float): SLIC param, width of Gaussian smoothing kernel for pre-processing for each
			dimension of the image. Zero means no smoothing.

		model (src.models.autoencoder3d.AutoEncoder3D | None): the AutoEncoder3D model.
			If None the resulting graph will contain only the percentile intensity features.
		saved_path (str | None): folder path where the pretrained model is saved.

		percentiles (list): list of intenisties percentiles to compute. (See get_node_percentiles_features() function).

		k (int): number of edges neighbors to keep. (See get_node_edges() function).

		write_to_file (str | None): Wheather save the resulting graph to file. If not None
			a valid folder path where to save files must be specified.

	Returns:
		graph (torch_geometric.data.data.Data): the graph representing a 4D multichannel image.
		centroids (numpy.ndarray): the graph nodes centroids.
	"""
	base_image_transform, _, autoencoder_eval_transform, _, _ = get_transformations()
	image_trans = base_image_transform(data)
	supervoxels, stacked_volume = get_supervoxels(
		multi_channel_volume = image_trans['image'],
		n_segments = n_segments,
		compactness = compactness,
		sigma = sigma
	)
	if not model is None:
		_, latent_maps, _ = predict_ae(
			model = model,
			data = data,
			transforms = autoencoder_eval_transform,
			device = 'cpu', # get_device(), # 'mps' not supported
			saved_path = saved_path
		)
	node_features, centroids, labels, voxels = [], [], [], []
	for _, region in enumerate(regionprops(supervoxels)):
		supervoxel_mask = (supervoxels == region.label)
		supervoxel_features = get_node_percentiles_features(
			supervoxel_mask = supervoxel_mask,
			stacked_volume = stacked_volume,
			percentiles = percentiles
		)
		# Filters out supervoxels located outside the brain mass, characterized by zero intensity
		if np.array(supervoxel_features).any():
			if not model is None:
				stride = [2, 2, 2]
				bbox = region.bbox
				latent_bbox = [
					max(0, bbox[0] // stride[0]),
					max(0, bbox[1] // stride[1]),
					max(0, bbox[2] // stride[2]),
					min(latent_maps.shape[1], (bbox[3] + 1) // stride[0]),
					min(latent_maps.shape[2], (bbox[4] + 1) // stride[1]),
					min(latent_maps.shape[3], (bbox[5] + 1) // stride[2])
				]
				latent_patch = latent_maps[:,
					latent_bbox[0]:latent_bbox[3],
					latent_bbox[1]:latent_bbox[4],
					latent_bbox[2]:latent_bbox[5]
				]
				fixed_size = (3, 3, 3)
				latent_patch_resized = F.adaptive_avg_pool3d(latent_patch.unsqueeze(0), fixed_size).squeeze(0)
				bn = np.array(latent_patch_resized).mean(axis=(0)).flatten()
				_, S, _ = np.linalg.svd(bn.reshape(9, 3))
				latent_features = np.concatenate([S, bn]).reshape((len(S) + len(bn)))
				node_features.append(np.concatenate([supervoxel_features, latent_features]).reshape((len(supervoxel_features) + len(latent_features))))
			else:
				node_features.append(supervoxel_features)
			voxels.append(np.argwhere(supervoxel_mask))
			centroids.append(region.centroid)
			labels.append(get_node_labels(supervoxel_mask, image_trans['label']))
	centroids = np.array(centroids)
	graph = Data(
		x = torch.tensor(node_features, dtype=torch.float),
		edge_index = torch.tensor(get_node_edges(centroids = centroids, k = k).T, dtype=torch.float),
		y = torch.tensor(labels, dtype=torch.float)
	)

	# Save data to file
	if write_to_file:
		fold = get_config().get('GRAPH_FOLDER')
		if not os.path.isdir(os.path.join(write_to_file, fold)):
			os.makedirs(os.path.join(write_to_file, fold))
		if not os.path.isdir(os.path.join(write_to_file, fold, data['subject'])):
			os.makedirs(os.path.join(write_to_file, fold, data['subject']))
		torch.save(graph, os.path.join(write_to_file, fold, data['subject'], (data['subject'] + '.graph')))
		torch.save(voxels, os.path.join(write_to_file, fold, data['subject'], (data['subject'] + '.map')))
	return graph, centroids


def get_supervoxels(
		multi_channel_volume,
		n_segments = 15000,
		compactness = .1,
		sigma = 1
	):
	"""
	Given a multichannel 4D MRI image, applys the SLIC algorithm to obtain supervoxels.
	Args:
		multi_channel_volume (monai.data.meta_tensor.MetaTensor | numpy.ndarray): the 4D multichannel
			input image.
		n_segments (int): the (approximate) number of labels in the segmented output image.
		compactness (float): balances color proximity and space proximity. Higher values give
			more weight to space proximity, making superpixel shapes more square/cubic.
		sigma (float): width of Gaussian smoothing kernel for pre-processing for each
			dimension of the image. Zero means no smoothing.
	Returns:
		supervoxels (numpy.ndarray): the SLIC output, 3D supervoxels image.
		stacked_volume (numpy.ndarray): the stacked 4-channel image.
	"""
	stacked_volume = np.stack(multi_channel_volume, axis=-1)
	supervoxels = slic(
		stacked_volume,
		n_segments = n_segments,
		compactness = compactness,
		sigma = sigma,
		start_label = 0,
		channel_axis = 3,
		convert2lab = False
	)
	return supervoxels, stacked_volume


def get_node_percentiles_features(
		supervoxel_mask,
		stacked_volume,
		percentiles = [10, 25, 50, 75, 90]
	):
	"""
	Computes the percentiles features by supervoxel.
	Args:
		supervoxel_mask (numpy.ndarray): binary mask of supervoxel.
		stacked_volume (numpy.ndarray): the stacked 4-channel image.
		percentiles (list): list of intenisties percentiles to compute.
	Returns:
		supervoxel_features (list): the percentiles features extracted by supervoxel.
	"""
	supervoxel_features = []
	for c in range(stacked_volume.shape[-1]):
		voxel_values = stacked_volume[..., c][supervoxel_mask]
		channel_percentiles = np.percentile(voxel_values, percentiles)
		supervoxel_features.extend(channel_percentiles)
	return supervoxel_features


def get_node_labels(
		supervoxel_mask,
		truth
	):
	"""
	Computes the labels by supervoxel.
	Args:
		supervoxel_mask (numpy.ndarray): binary mask of supervoxel.
		truth (monai.data.meta_tensor.MetaTensor | numpy.ndarray): the 4D multichannel label image.
	Returns:
		node_label (int): the label by supervoxel. One supervoxel can be assigned to
			more than one class according to BraTS-2023 classes:
			- label 0 (NT): the non-tumorous area.
			- label 1 (NCR): the necrotic tumor core.
			- label 2 (ED): the peritumoral edematous/invaded tissue.
			- label 3 (ET): the GD-enhancing tumor.
	"""
	voxel_labels = (truth[0][supervoxel_mask]).astype(np.int64)
	return np.bincount(voxel_labels).argmax()


def get_node_edges(centroids, k=10):
	"""
	Computes the graph edges.
	Args:
		centroids (numpy.ndarray): the graph centroid list.
		k (int): number of neighbor edges to keep.
	Returns:
		edges (list): the graph edges.
	"""
	_, indices = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(centroids).kneighbors(centroids)
	return np.array([[i, j] for i in range(indices.shape[0]) for j in indices[i, 1:]])
