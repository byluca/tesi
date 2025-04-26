"""
A set of plotting functions
"""
import os, random, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn import plotting
import nibabel as nib
from src.helpers.utils import get_colored_mask, get_slice, get_brats_classes
from src.helpers.config import get_config
import plotly.graph_objects as go
from torch_geometric.utils import to_networkx
import networkx as nx


__all__ = ['random_samples', 'single_sample', 'counter', 'brats_classes', 'before_after', 'training_values', 'graph', 'prediction', 'hp_ranges', 'best_config']


def random_samples(folder, n_samples, axis):
	"""
	Plot a comparison between different n data examples.
	Args:
		folder (str): the path of the folder containing data.
		n_samples (int): number of samples to plot. Min value is 2.
		axis (int): axis of the spatial image. Values are: 0=X_axis, 1=Y_axis, 2=Z_axis.
	Returns:
		None.
	"""
	_config = get_config()
	channels = _config.get('CHANNELS')
	if n_samples > 1 and axis < 3:
		samples = random.sample(os.listdir(folder), n_samples)
		fig, axs = plt.subplots(n_samples, 5, figsize=(18, n_samples * 4))
		for i, axl in enumerate(axs):
			images = sorted(os.listdir(os.path.join(folder, samples[i])))
			images.append(images.pop(images.index(images[0])))
			if i == 0:
				for k in range(5):
					axl[k].set_title(channels[k])
			for j, ax in enumerate(axl):
				if j == 0:
					ax.set_ylabel(samples[i])
				brain_vol = nib.load(os.path.join(os.path.join(folder, samples[i]), images[j]))
				isocenter = list(map(int, plotting.find_xyz_cut_coords(brain_vol)))
				if j == len(images) - 1:
					mask_colored, _ = get_colored_mask(get_slice(brain_vol, axis, isocenter[axis]))
					brain_vol = nib.load(os.path.join(os.path.join(folder, samples[i]), images[j - 1]))
					ax.imshow(get_slice(brain_vol, axis, isocenter[axis]), cmap='gist_yarg')
					ax.imshow(mask_colored)
				else:
					ax.imshow(get_slice(brain_vol, axis, isocenter[axis]), cmap='gist_yarg')
				ax.spines['top'].set_visible(False)
				ax.spines['right'].set_visible(False)
				ax.spines['bottom'].set_visible(False)
				ax.spines['left'].set_visible(False)
				a = {0:'X',1:'Y',2:'Z'}
				ax.set_xlabel(a[axis]+'='+str(isocenter[axis]))
				ax.get_xaxis().set_ticks([])
				ax.get_yaxis().set_ticks([])
		fig.tight_layout()
		plt.show()
	else:
		print('\n' + ''.join(['> ' for i in range(40)]))
		print('\nERROR: \033[95m n_samples\033[0m must be greater that \033[95m 1\033[0m and \033[95m axis\033[0m bust be lower that \033[95m 2\033[0m.\n')
		print(''.join(['> ' for i in range(40)]) + '\n')


def single_sample(folder, session=None):
	"""
	Plot different views of a single sample data.
	Args:
		folder (str): the path of the folder containing data.
		session (str): a session ID, if not provided a random ID will be selected.
	Returns:
		None.
	"""
	_config = get_config()
	channels = _config.get('CHANNELS')
	sample = session if session else random.sample(os.listdir(folder), 1)[0]
	images = sorted(os.listdir(os.path.join(folder, sample)))
	images.append(images.pop(images.index(images[0])))
	fig, axs = plt.subplots(3, 5, figsize=(18, 12))
	for i, axl in enumerate(axs):
		if i == 0:
			for k in range(5):
				axl[k].set_title(channels[k])
		for j, ax in enumerate(axl):
			brain_vol = nib.load(os.path.join(os.path.join(folder, sample), images[j]))
			isocenter = list(map(int, plotting.find_xyz_cut_coords(brain_vol)))
			if j == len(images) - 1:
				mask_colored, _ = get_colored_mask(get_slice(brain_vol, i, isocenter[i]))
				brain_vol = nib.load(os.path.join(os.path.join(folder, sample), images[j - 1]))
				ax.imshow(get_slice(brain_vol, i, isocenter[i]), cmap='gist_yarg')
				ax.imshow(mask_colored)
			else:
				ax.imshow(get_slice(brain_vol, i, isocenter[i]), cmap='gist_yarg')
			ax.spines['top'].set_visible(False)
			ax.spines['right'].set_visible(False)
			ax.spines['bottom'].set_visible(False)
			ax.spines['left'].set_visible(False)
			a = {0:'X',1:'Y',2:'Z'}
			ax.set_xlabel(a[i]+'='+str(isocenter[i]))
			ax.get_xaxis().set_ticks([])
			ax.get_yaxis().set_ticks([])
	brain_vol = nib.load(os.path.join(os.path.join(folder, sample), images[j - 1]))
	brain_mask = nib.load(os.path.join(os.path.join(folder, sample), images[j]))
	plotting.plot_epi(brain_vol, display_mode='z', cmap='hot_white_bone', title=images[j - 1])
	plotting.plot_epi(brain_mask, display_mode='z', cmap='hot_white_bone', title=images[j])
	fig.tight_layout()
	fig.suptitle(sample, fontsize=18)
	plt.show()
	return sample


def counter(folder):
	"""
	Plot data counters.
	Args:
		folder (str): the path of the folder containing data.
	Returns:
		None.
	"""
	mr_sessions = sorted([f.split('-')[2] for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))])
	n_subjects = len(list(set(mr_sessions)))
	mr_sessions = len(mr_sessions)
	fig, ax = plt.subplots(1, 1, figsize=(18, 8))
	bar_labels = ['MRI Sessions (n.'+str(mr_sessions)+')', 'Nr. Subjects (n.'+str(n_subjects)+')']
	bars = ax.bar(['MRI Sessions', 'Nr. Subjects'], height=[mr_sessions, n_subjects], label=bar_labels, color=['#8fce00', '#ff8200'])
	for rect in bars:
		height = rect.get_height()
		plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.0f} ({height/mr_sessions*100:.2f}%)', ha='center', va='bottom')
	plt.xlabel('DATA', labelpad=20)
	plt.ylabel('COUNT', labelpad=20)
	plt.title('EXPLORING DATA')
	plt.legend()
	fig.tight_layout()
	plt.show()


def brats_classes(folder, session=None, axis=2):
	"""
	Plot data labels and data classes according to BraTS-2023.
	Args:
		folder (str): the path of the folder containing data.
		session (str): a session ID, if not provided a random ID will be selected.
		axis (int): axis of the spatial image. Values are: 0=X_axis, 1=Y_axis, 2=Z_axis.
	Returns:
		None.
	"""
	_config = get_config()
	labels = _config.get('LABELS')
	classes = _config.get('CLASSES')
	sample = session if session else random.sample(os.listdir(folder), 1)[0]
	images = sorted(os.listdir(os.path.join(folder, sample)))
	fig, axs = plt.subplots(2, 3, figsize=(18, 12))
	brain_mask = nib.load(os.path.join(os.path.join(folder, sample), images[0]))
	brain_vol = nib.load(os.path.join(os.path.join(folder, sample), images[1]))
	print(f"Image shape: {brain_vol.shape}")
	isocenter = list(map(int, plotting.find_xyz_cut_coords(brain_mask)))
	_, label_masks = get_colored_mask(get_slice(brain_mask, axis, isocenter[axis]))
	brain_classes = get_brats_classes(brain_mask)
	for i, axl in enumerate(axs):
		for j, ax in enumerate(axl):
			if i == 0:
				ax.imshow(get_slice(brain_vol, axis, isocenter[axis]), cmap='gist_yarg')
				ax.imshow(label_masks[j])
				ax.set_title('Label '+str(j+1)+': '+labels[j])
			else:
				ax.imshow(get_slice(brain_classes[j], axis, isocenter[axis]), cmap='gray')
				ax.set_title('Class '+str(j+1)+': '+classes[j])
			ax.axis('off')
	fig.tight_layout()
	plt.show()


def before_after(before, after, titles):
	"""
	Plot data comparing two images
		(i.e. before and after preprocessing or original and AE reconstructed images).
	Args:
		before (monai.data.meta_tensor.MetaTensor | list): list of image paths for each modatlity.
		after (monai.data.meta_tensor.MetaTensor | numpy.ndarray): the 4D multichannel input image.
		titles (list): plot titles.
	Returns:
		None.
	"""
	_config = get_config()
	channels = _config.get('CHANNELS')
	after_vol = after.detach().cpu().numpy()
	if type(before) is list:
		plt.figure('before', (18, 6))
		for i in range(4):
			plt.subplot(1, 4, i + 1)
			plt.title(channels[i])
			plt.axis('off')
			brain_vol = nib.load(before[i])
			isocenter_vol = list(map(int, plotting.find_xyz_cut_coords(brain_vol)))
			plt.imshow(get_slice(brain_vol, 2, isocenter_vol[2]), cmap='gray')
		plt.suptitle(titles[0], fontsize=18, y=0.88)
		plt.show()
	else:
		before_vol = after.detach().cpu().numpy()
		plt.figure('before', (18, 6))
		for i in range(4):
			plt.subplot(1, 4, i + 1)
			plt.title(channels[i])
			plt.axis('off')
			brain_vol3d = nib.Nifti1Image(before_vol[i], affine=np.eye(4))
			isocenter_vol = list(map(int, plotting.find_xyz_cut_coords(brain_vol3d)))
			plt.imshow(get_slice(before_vol[i], 2, isocenter_vol[2]), cmap='gray')
		plt.suptitle(titles[0], fontsize=18, y=0.88)
		plt.show()
	plt.figure('after', (18, 6))
	for i in range(4):
		plt.subplot(1, 4, i + 1)
		plt.title(channels[i])
		plt.axis('off')
		brain_vol3d = nib.Nifti1Image(after_vol[i], affine=np.eye(4))
		isocenter_vol = list(map(int, plotting.find_xyz_cut_coords(brain_vol3d)))
		plt.imshow(get_slice(after_vol[i], 2, isocenter_vol[2]), cmap='gray')
	plt.suptitle(titles[1], fontsize=18, y=0.88)
	plt.show()


def training_values(folder, model_ids):
	"""
	Plot losses and metrics over training phase.
	Args:
		folder (str): the path of the folder containing the csv reports.
		model_ids (list): the model run ids to load its relative training values.
	Returns:
		None.
	"""
	for run_id in model_ids:
		data_path = os.path.join(folder, run_id.split('_')[0] + '_training.csv')
		if os.path.isfile(data_path):
			best_metric = 'metric_ssim' if 'AUTOENCODER3D' in run_id else 'metric_dice_avg'
			fig, ax_row = plt.subplots(1, 2, figsize=(18, 6))
			df = pd.read_csv(data_path)
			data_df = df[df['id'] == run_id]
			if len(data_df):
				best_epoch = df.iloc[data_df[best_metric].idxmax()]['epoch']
				x = [i + 1 for i in range(len(data_df))]
				ax_row[0].plot(x, data_df['train_loss'].to_numpy(), label='training_loss')
				ax_row[0].plot(x, data_df['eval_loss'].to_numpy(), label='evaluation_loss')
				ax_row[0].set_xticks([i for i in range(0, len(data_df), (5 if len(data_df) <= 100 else 20))])
				ax_row[0].axvline(best_epoch, color='red')
				ax_row[0].text(best_epoch - 3.2, data_df['train_loss'].max() / 2, 'best_run', rotation=0)
				ax_row[0].set_xlabel('EPOCHS', fontsize=14)
				ax_row[0].set_ylabel('LOSSES', fontsize=14)
				ax_row[0].set_title(run_id.split('_')[0], fontsize=18)
				ax_row[0].legend(loc='upper center')
				if 'AUTOENCODER3D' in run_id:
					ax_row[1].plot(x, data_df['metric_ssim'].to_numpy(), label='SSIM (Structural Similarity Index Metric)')
					ax_row[1].plot(x, data_df['metric_psnr'].to_numpy() / 100, label='PSNR / 100 (Peak Signal-to-Noise Ratio)')
					ax_row[1].plot(x, data_df['metric_ncc'].to_numpy(), label='NCC (Normalized Cross-Correlation)')
				else:
					ax_row[1].plot(x, data_df['metric_dice_avg'].to_numpy(), label='Avg. Dice Score')
				ax_row[1].set_xticks([i for i in range(0, len(data_df), (5 if len(data_df) <= 100 else 20))])
				ax_row[1].set_yticks(np.round(np.linspace(.0, 1., 10), 1))
				ax_row[1].axvline(best_epoch, color='red')
				ax_row[1].text(best_epoch - 3.2, data_df['train_loss'].max() / 2, 'best_run', rotation=0)
				ax_row[1].set_xlabel('EPOCHS', fontsize=14)
				ax_row[1].set_ylabel('SCORES', fontsize=14)
				ax_row[1].set_title(run_id.split('_')[0], fontsize=18)
				ax_row[1].legend(loc='lower center')
				fig.tight_layout()
				plt.show()
			else:
				print('\n' + ''.join(['> ' for i in range(30)]))
				print('\nERROR: no data found for \033[95m' + run_id + '\033[0m run id.\n')
				print(''.join(['> ' for i in range(30)]) + '\n')
		else:
			print('\n' + ''.join(['> ' for i in range(30)]))
			print('\nERROR: no model report found.\n')
			print(''.join(['> ' for i in range(30)]) + '\n')


def graph(graph, centroids):
	"""
	Plot a static and interactive version of a graph data entry.
	Args:
		graph (torch_geometric.data.data.Data): the graph data structure.
		centroids (numpy.ndarray): the centroid list for each node in the graph.
	Returns:
		None.
	"""
	fig = plt.figure(figsize=(18, 12))
	ax = fig.add_subplot(111, projection='3d')
	brain = np.array([c for i, c in enumerate(centroids) if not graph.y[i].any()])
	wt = np.array([c for i, c in enumerate(centroids) if graph.y[i] != .0])
	ax.scatter(brain[:, 0], brain[:, 1], brain[:, 2], c='blue', s=4, label='No Tumor')
	ax.scatter(wt[:, 0], wt[:, 1], wt[:, 2], c='red', s=8, label='Whole Tumor')
	edges = graph.edge_index.T.tolist()
	for src, dest in edges:
		src, dest = int(src), int(dest)
		ax.plot(
			[centroids[src, 0], centroids[dest, 0]],
			[centroids[src, 1], centroids[dest, 1]],
			[centroids[src, 2], centroids[dest, 2]],
			c='gray',
			linewidth=0.08
		)
	plt.legend(loc='upper right')
	plt.title('3D Static Graph Visualization')
	plt.show()
	nx_graph = to_networkx(graph)
	pos = nx.spring_layout(nx_graph, dim=3)
	label_colors = {'No Tumor': 'blue', 'Whole Tumor': 'red'}
	grouped_nodes = {label: [] for label in label_colors.keys()}
	for node, label in enumerate(graph.y.tolist()):
		label = 'No Tumor' if label == .0 else 'Whole Tumor'
		grouped_nodes[label].append(node)
	edge_x, edge_y, edge_z = [], [], []
	for edge in nx_graph.edges():
		x0, y0, z0 = pos[int(edge[0])]
		x1, y1, z1 = pos[int(edge[1])]
		edge_x.extend([x0, x1, None])
		edge_y.extend([y0, y1, None])
		edge_z.extend([z0, z1, None])
	edge_trace = go.Scatter3d(
		x=edge_x, y=edge_y, z=edge_z,
		mode='lines',
		line=dict(color='gray', width=.6),
		hoverinfo='none',
		showlegend=False
	)
	node_traces = []
	for label, nodes in grouped_nodes.items():
		x_nodes = [pos[n][0] for n in nodes]
		y_nodes = [pos[n][1] for n in nodes]
		z_nodes = [pos[n][2] for n in nodes]
		node_traces.append(
			go.Scatter3d(
				x=x_nodes, y=y_nodes, z=z_nodes,
				mode='markers',
				marker=dict(size=3, color=label_colors[label]),
				name=label,
				hovertext=[f'Node {n} - {label}' for n in nodes],
				hoverinfo='text',
			)
		)
	fig = go.Figure(data=[edge_trace] + node_traces)
	fig.update_layout(
		width=1200,
		height=900,
		title='3D Interactive Graph Visualization',
		legend=dict(title='Node Labels', x=0.8, y=0.9),
	)
	fig.show()


def prediction(data):
	"""
	Plot original image data, groundtruth label and predicted mask.
	Args:
		data (list): list containing the original 4D MRI, the 3D ground truth
			and the 3D predicted mask ad numpy.ndarray.
	Returns:
		None.
	"""
	_config = get_config()
	classes = _config.get('CLASSES')
	channels = _config.get('CHANNELS')
	data_str = {'image': data[0], 'label': data[1], 'pred': data[2]}
	for i, k in enumerate(data_str.keys()):
		img = get_brats_classes(data_str[k][0])[0] if k == 'label' else data_str[k][0]
		img = img.astype(np.float32)
		brain_vol3d = nib.Nifti1Image(img, affine=np.eye(4))
		isocenter = list(map(int, plotting.find_xyz_cut_coords(brain_vol3d)))
		plt.figure('image', (18, 6))
		for j in range(4 if i == 0 else 3):
			plt.subplot(1, 4 if i == 0 else 3, j + 1)
			plt.title(k + '_' + (channels[j] if i == 0 else classes[j]))
			plt.axis('off')
			if k == 'label':
				img = get_brats_classes(data_str[k][0])[j, :, :, isocenter[2]]
				plt.imshow(np.rot90(img, 0), cmap = 'viridis')
			else:
				img = data_str[k][j, :, :, isocenter[2]]
				plt.imshow(img, cmap = ('gray' if i == 0 else 'viridis'))
		plt.show()


def hp_ranges(models = []):
	"""
	Plot the ranges for each hyperparameter by grid-searching and for each model.
		If no model is passed the default GNNs will be considered.
	Args:
		models (list): list with model names.
	Returns:
		None.
	"""
	_config = get_config()
	models = models if len(models) else _config.get('GNN')
	for m in models:
		param_grid = _config.get('HP_TUNING')['SHARED'] | _config.get('HP_TUNING')[m]
		combinations = list(itertools.product(*param_grid.values()))
		print(''.join(['> ' for i in range(40)]))
		print(f'\033[1m{"MODEL":<18}{m.upper():>35}{"RUNs":>10}{str(len(combinations)):>10}\033[0m')
		for k in param_grid.keys():
			if k == 'hidden_channels':
				for i, v in enumerate(param_grid[k]):
					print(f'{(k if i == 0 else ""):<18}{str(v):>35}')
			else:
				print(f'{k:<18}{str(param_grid[k]):>35}')


def best_config(reports_path, models = []):
	"""
	Plot the best hyperparameters values as found by grid-search for each model.
		If no model is passed the default GNNs will be considered.
		A valid csv report file must be found in reports folder.
	Args:
		reports_path (str): the folder path where to read a valid csv report file.
		models (list): list with model names.
	Returns:
		best_run (list): list of the best run for each model found in `reports_path`.
	"""
	_config = get_config()
	models = models if len(models) else _config.get('GNN')
	best_run = []
	for m in models:
		p = os.path.join(reports_path, m + '_tuning.csv')
		if os.path.isfile(p):
			print(''.join(['> ' for i in range(40)]))
			df = pd.read_csv(p, encoding='UTF-8')
			best_row = df.iloc[df['node_dice_score'].idxmax()].drop(['model', 'n_images', 'exec_time', 'datetime'], axis=0)
			for i, k in enumerate(best_row.keys()):
				if i == 0:
					best_run.append(best_row[k])
					print(f'\033[1m{k:<18}{str(best_row[k]):>35}\033[0m')
				else:
					print(f'{k:<18}{str(best_row[k]):>35}')
	return best_run
