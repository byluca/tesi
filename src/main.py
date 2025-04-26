"""
Executable script to train and test a model.
	Args:
	--model (str): name of the model, case sensitive (i.e., `--model=GraphSAGE`).
		Possible values: `AutoEncoder3D`, `GraphSAGE`, `GAT`, `ChebNet`.
"""
import os, sys, random
from sys import platform
_base_path = '\\'.join(os.getcwd().split('\\')) + '\\' if platform == 'win32' else '/'.join(os.getcwd().split('/')) + '/'
sys.path.append(_base_path)
from monai.utils import set_determinism
from src.helpers.utils import make_dataset, get_device
from src.modules.training import train_test_splitting, training_model, predict_gnn
from src.helpers.config import get_config
from src.modules.preprocessing import get_transformations
from src.models.autoencoder3d import AutoEncoder3D
from src.models.gnn import GraphSAGE, GAT, ChebNet


# defining paths
_config = get_config()
data_path = os.path.join(_base_path, _config.get('DATA_FOLDER'))
graph_path = os.path.join(data_path, _config.get('GRAPH_FOLDER'))
saved_path = os.path.join(_base_path, _config.get('SAVED_FOLDER'))
reports_path = os.path.join(_base_path, _config.get('REPORT_FOLDER'))
logs_path = os.path.join(_base_path, _config.get('LOG_FOLDER'))
if platform == 'win32':
	data_path = data_path.replace('/', '\\')
	graph_path = graph_path.replace('/', '\\')
	saved_path = saved_path.replace('/', '\\')
	reports_path = reports_path.replace('/', '\\')
	logs_path = logs_path.replace('/', '\\')


# defining default settings
num_node_features = 20			# Input feature size
num_classes = 4					# Number of output classes
lr = 1e-4						# Learning rate for the optimizier
weight_decay = 1e-5				# Weight decay for the optimizier
dropout = .1					# Dropout probability (for features)
hidden_channels = [512, 512, 512, 512, 512, 512, 512] # No. of hidden units (input layer included output layer excluded)
# GRAPHSAGE PARAMS
aggr = 'mean'				# Apply pooling operation as aggregator
# GAT PARAMS
heads = 14					# Number of attention heads
attention_dropout = .2		# Dropout probability (for attention mechanism)
# CHEBNET PARAMS
k = 3						# Chebyshev polynomial order


# defining models
_models = {
	'AutoEncoder3D': AutoEncoder3D(
		spatial_dims=3,
		in_channels=4,
		out_channels=4,
		channels=(5,),
		strides=(2,),
		inter_channels=(8, 8, 16),
		inter_dilations=(1, 2, 4)
	),
	'GraphSAGE': GraphSAGE(
		in_channels = num_node_features,
		hidden_channels = hidden_channels,
		out_channels = num_classes,
		dropout = dropout,
		aggr = aggr
	),
	'GAT': GAT(
		in_channels = num_node_features,
		hidden_channels = hidden_channels,
		out_channels = num_classes,
		dropout = dropout,
		heads = heads,
		attention_dropout = attention_dropout
	),
	'ChebNet': ChebNet(
		in_channels = num_node_features,
		hidden_channels = hidden_channels,
		out_channels = num_classes,
		dropout = dropout,
		K = k
	)
}


if __name__ == "__main__":

	args = sys.argv[1:]
	if len(args) == 0:
		print('\n' + ''.join(['> ' for i in range(25)]))
		print('\nWARNING: missing required parameters!')
		print('\n' + ''.join(['> ' for i in range(25)]))
		print(f'\n{"PARAM":<16}{"VALUE RANGE":<18}\n')
		print(''.join(['> ' for i in range(25)]))
		print(f'{"--model":<16}{str(list(_models.keys())):<18}')
		print(''.join(['> ' for i in range(25)]) + '\n')
	else:
		keys = [i.split('=')[0].upper()[2:] for i in args]
		values = [i.split('=')[1] for i in args]
		model_name = values[keys.index('MODEL')]

		if model_name in _models.keys():

			# ensure reproducibility
			set_determinism(seed=3)
			random.seed(3)

			# set model
			model = _models[model_name]

			print(get_device())

			# load and splitting data
			if model.name != 'AutoEncoder3D':
				data_folder = make_dataset(dataset='glioma', verbose=False, base_path=_base_path)
				train_data, eval_data, test_data = train_test_splitting(data_folder, reports_path=reports_path, load_from_file=True)
			else:
				data_folder = os.path.join(data_path, 'tumor-mix', 'BraTS-PDGM')
				train_d, eval_d, test_d = train_test_splitting(data_folder, reports_path=reports_path, load_from_file=True)
				train_data = train_d + eval_d
				eval_data = test_d

			# get data transformations pipelines
			(
				_,
				autoencoder_train_transform,
				autoencoder_eval_transform,
				gnn_train_eval_transform,
				gnn_test_transform
			) = get_transformations(graph_path)
			transforms = [autoencoder_train_transform, autoencoder_eval_transform] if model.name == 'AutoEncoder3D' else [gnn_train_eval_transform, gnn_train_eval_transform]

			# training model
			_ = training_model(
				model = model,
				data = [train_data, eval_data],
				transforms = transforms,
				epochs = 100 if model.name == 'AutoEncoder3D' else 500,
				device = get_device(),
				paths = [saved_path, reports_path, logs_path, graph_path],
				early_stopping = 10 if model.name == 'AutoEncoder3D' else 25,
				ministep = 14 if model.name == 'AutoEncoder3D' else 6,
				lr = 1e-4 if model.name == 'AutoEncoder3D' else lr,
				weight_decay = 1e-5 if model.name == 'AutoEncoder3D' else weight_decay
			)

			# testing model
			if model.name != 'AutoEncoder3D':
				_, _ = predict_gnn(
					model = model,
					data = test_data,
					transforms = gnn_test_transform,
					device = get_device(),
					paths = [saved_path, reports_path, logs_path],
					return_predictions = False,
					verbose = True
				)

		else:
			print('\n' + ''.join(['> ' for i in range(25)]))
			print('\nWARNING: out-of-bound parameters!')
			print('\n' + ''.join(['> ' for i in range(25)]))
			print(f'\n{"PARAM":<16}{"VALUE RANGE":<18}\n')
			print(''.join(['> ' for i in range(25)]))
			print(f'{"--model":<16}{str(list(_models.keys())):<18}')
			print(''.join(['> ' for i in range(25)]) + '\n')

	sys.exit(0)