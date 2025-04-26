"""
Executable script for hyperparameter tuning with grid search.
	For hyperparameter ranges please refer to `src.helpers.config`.
	Args:
	--model (str): name of the model, case sensitive (i.e., `--model=GraphSAGE`).
		Possible values: `GraphSAGE`, `GAT`, `ChebNet`.
"""
import os, sys, random, itertools, calendar, time
from sys import platform
_base_path = '\\'.join(os.getcwd().split('\\')) + '\\' if platform == 'win32' else '/'.join(os.getcwd().split('/')) + '/'
sys.path.append(_base_path)
import pandas as pd
from monai.utils import set_determinism
from src.helpers.utils import make_dataset, get_device, get_date_time, save_results
from src.modules.training import train_test_splitting, training_model, predict_gnn
from src.helpers.config import get_config
from src.modules.preprocessing import get_transformations
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
num_node_features = 20		# Number of input feature by node
num_classes = 4				# Number of output classes


if __name__ == "__main__":

	args = sys.argv[1:]
	if len(args) == 0:
		print('\n' + ''.join(['> ' for i in range(25)]))
		print('\nWARNING: missing required parameters!')
		print('\n' + ''.join(['> ' for i in range(25)]))
		print(f'\n{"PARAM":<16}{"VALUE RANGE":<18}\n')
		print(''.join(['> ' for i in range(25)]))
		print(f'{"--model":<16}{str(list(_config.get("HP_TUNING").keys())[1:]):<18}')
		print(''.join(['> ' for i in range(25)]) + '\n')
	else:
		keys = [i.split('=')[0].upper()[2:] for i in args]
		values = [i.split('=')[1] for i in args]
		model_name = values[keys.index('MODEL')]

		if model_name in list(_config.get('HP_TUNING').keys())[1:]:

			# ensure reproducibility
			set_determinism(seed=3)
			random.seed(3)

			# load and splitting data
			data_folder = make_dataset(dataset='glioma', verbose=False, base_path=_base_path)
			train_data, eval_data, test_data = train_test_splitting(data_folder, reports_path=reports_path, load_from_file=True)

			# get data transformations pipelines
			(_, _, _, gnn_train_eval_transform, gnn_test_transform) = get_transformations(graph_path)

			# set model and parameters ranges for grid searching
			param_grid = _config.get('HP_TUNING')['SHARED'] | _config.get('HP_TUNING')[model_name]
			combinations = list(itertools.product(*param_grid.values()))
			for n, config in enumerate(combinations):
				run_id = model_name.upper() + '_' + str(calendar.timegm(time.gmtime()))
				file_report = os.path.join(reports_path, model_name.upper() + '_tuning.csv')
				lr, weight_decay, dropout, hidden_channels = config[0], config[1], config[2], config[3]
				if model_name == 'GraphSAGE':
					aggr = config[4]
					model = GraphSAGE(
						in_channels = num_node_features,
						hidden_channels = hidden_channels,
						out_channels = num_classes,
						dropout = dropout,
						aggr = aggr
					)
					if not os.path.isfile(file_report):
						template = pd.DataFrame({'id':[],'model':[],'n_images':[],'node_dice_score':[],'exec_time':[],'lr':[],'weight_decay':[],'dropout':[],'hidden_channels':[],'aggr':[],'datetime':[]})
						template.to_csv(file_report, encoding = 'UTF-8', index = False)
					df = pd.read_csv(file_report, encoding = 'UTF-8')
					df = df[(df['lr'] == lr) & (df['weight_decay'] == weight_decay) & (df['dropout'] == dropout) & (df['hidden_channels'] == str(hidden_channels)) & (df['aggr'] == aggr)]
				elif model_name == 'GAT':
					heads, attention_dropout = config[4], config[5]
					model = GAT(
						in_channels = num_node_features,
						hidden_channels = hidden_channels,
						out_channels = num_classes,
						dropout = dropout,
						heads = heads,
						attention_dropout = attention_dropout
					)
					if not os.path.isfile(file_report):
						template = pd.DataFrame({'id':[],'model':[],'n_images':[],'node_dice_score':[],'exec_time':[],'lr':[],'weight_decay':[],'dropout':[],'hidden_channels':[],'heads':[],'attention_dropout':[],'datetime':[]})
						template.to_csv(file_report, encoding = 'UTF-8', index = False)
					df = pd.read_csv(file_report, encoding = 'UTF-8')
					df = df[(df['lr'] == lr) & (df['weight_decay'] == weight_decay) & (df['dropout'] == dropout) & (df['hidden_channels'] == str(hidden_channels)) & (df['heads'] == heads) & (df['attention_dropout'] == attention_dropout)]
				else:
					k = config[4]
					model = ChebNet(
						in_channels = num_node_features,
						hidden_channels = hidden_channels,
						out_channels = num_classes,
						dropout = dropout,
						K = k
					)
					if not os.path.isfile(file_report):
						template = pd.DataFrame({'id':[],'model':[],'n_images':[],'node_dice_score':[],'exec_time':[],'lr':[],'weight_decay':[],'dropout':[],'hidden_channels':[],'k':[],'datetime':[]})
						template.to_csv(file_report, encoding = 'UTF-8', index = False)
					df = pd.read_csv(file_report, encoding = 'UTF-8')
					df = df[(df['lr'] == lr) & (df['weight_decay'] == weight_decay) & (df['dropout'] == dropout) & (df['hidden_channels'] == str(hidden_channels)) & (df['k'] == k)]

				if len(df) == 0:
					log = open(os.path.join(logs_path, 'hp_tuning.log'), 'a', encoding='utf-8')
					log.write('['+get_date_time() + '] EXECUTING.' + run_id + ' RUN ' + str(n + 1) + ' OF ' + str(len(combinations)) + ' \n')
					log.flush()

					# training model
					start = time.time()
					_ = training_model(
						model = model,
						data = [train_data, eval_data],
						transforms = [gnn_train_eval_transform, gnn_train_eval_transform],
						epochs = 500,
						device = get_device(),
						paths = [saved_path, reports_path, logs_path, graph_path],
						early_stopping = 25,
						ministep = 6,
						lr = lr,
						weight_decay = weight_decay,
						run_id = run_id
					)
					end = time.time()

					# testing model
					metrics, _ = predict_gnn(
						model = model,
						data = test_data,
						transforms = gnn_test_transform,
						device = get_device(),
						paths = [saved_path, reports_path, logs_path],
						return_predictions = False,
						write_to_file = False,
						verbose = True
					)

					dump = {
						'id': run_id,
						'model': model_name,
						'n_images': len(test_data),
						'node_dice_score': metrics[2][0],
						'exec_time': end - start,
						'lr': lr,
						'weight_decay': weight_decay,
						'dropout': dropout,
						'hidden_channels': hidden_channels
					}
					if model_name == 'GraphSAGE':
						dump['aggr'] = aggr
					elif model_name == 'GAT':
						dump['heads'] = heads
						dump['attention_dropout'] = attention_dropout
					else:
						dump['k'] = k
					dump['datetime'] = get_date_time()
					save_results(file = file_report, metrics = dump)
				else:
					log = open(os.path.join(logs_path, 'hp_tuning.log'), 'a', encoding='utf-8')
					log.write('['+get_date_time() + '] EXECUTED.' + run_id + ' RUN ' + str(n + 1) + ' OF ' + str(len(combinations)) + ' \n')
					log.flush()
			log.close()
		else:
			print('\n' + ''.join(['> ' for i in range(25)]))
			print('\nWARNING: out-of-bound parameters!')
			print('\n' + ''.join(['> ' for i in range(25)]))
			print(f'\n{"PARAM":<16}{"VALUE RANGE":<18}\n')
			print(''.join(['> ' for i in range(25)]))
			print(f'{"--model":<16}{str(list(_config.get("HP_TUNING").keys())[1:]):<18}')
			print(''.join(['> ' for i in range(25)]) + '\n')

	sys.exit(0)