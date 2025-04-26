"""
Parallelized program to preprocess the dataset and save it as a graph files.
Computes the node intensity features and (eventually) append to them
	the latent features extracted from the AutoEncoder3D.
Args:
	--process (int): number of processes to execute the target function in parallel.
		Must be in range between 1 and the maximun number of CPU cores.
		(i.e., `--process=4` this will run 4 process in parallel)
"""
import os, sys, random
from sys import platform
_base_path = '\\'.join(os.getcwd().split('\\')) + '\\' if platform == 'win32' else '/'.join(os.getcwd().split('/')) + '/'
sys.path.append(_base_path)
import numpy as np
from multiprocessing import Process, cpu_count
from monai.utils import set_determinism
from src.helpers.utils import make_dataset, get_date_time, verify_missing_writing
from src.modules.training import train_test_splitting
from src.helpers.config import get_config
from src.models.autoencoder3d import AutoEncoder3D
from src.helpers.graph import image_to_graph


# defining paths
_config = get_config()
data_path = os.path.join(_base_path, _config.get('DATA_FOLDER'))
saved_path = os.path.join(_base_path, _config.get('SAVED_FOLDER'))
reports_path = os.path.join(_base_path, _config.get('REPORT_FOLDER'))
logs_path = os.path.join(_base_path, _config.get('LOG_FOLDER'))
if platform == 'win32':
	data_path = data_path.replace('/', '\\')
	saved_path = saved_path.replace('/', '\\')
	reports_path = reports_path.replace('/', '\\')
	logs_path = logs_path.replace('/', '\\')


def target_function(data, model, proc_id):
	graph_path = _config.get('GRAPH_FOLDER')
	script = 'graph_constructor'
	log = open(os.path.join(logs_path, script + '.log'), 'a', encoding='utf-8')
	msg = '['+get_date_time()+'] Operation started.EXECUTING: PROCESS ' + str(proc_id) + '\n'
	print(msg), log.write(msg), log.flush()
	for i, d in enumerate(data):
		if not os.path.isfile(os.path.join(data_path, graph_path, d['subject'], (d['subject'] + '.graph'))):
			_, _ = image_to_graph(
				data = d,
				model = model,
				saved_path = saved_path,
				write_to_file = data_path
			)
		if i % int(len(data) / 12) == 0:
			msg = '['+get_date_time() + '] EXECUTING.PROCESS ' + str(proc_id) + ' SAMPLE ' + str((i if i > 0 else i+1)) + ' OF ' + str(len(data)) + ' \n'
			print(msg), log.write(msg), log.flush()
	msg = '['+get_date_time()+'] Operation ended.EXECUTING: PROCESS ' + str(proc_id) + '\n'
	print(msg),	log.write(msg), log.flush(), log.close()
	verify_missing_writing(data, os.path.join(data_path, graph_path), logs_path, script, proc_id)


if __name__ == "__main__":
	print('Your machine has ', cpu_count(), ' CPU cores.')
	args = sys.argv[1:]
	if len(args) == 0:
		print('\n' + ''.join(['> ' for i in range(25)]))
		print('\nWARNING: missing required parameters!')
		print('\n' + ''.join(['> ' for i in range(25)]))
		print(f'\n{"PARAM":<16}{"VALUE RANGE":<18}\n')
		print(''.join(['> ' for i in range(25)]))
		print(f'{"--process":<16}{("min: 1 and max: " + str(cpu_count()) + "."):<18}')
		print(''.join(['> ' for i in range(25)]) + '\n')
	else:
		keys = [i.split('=')[0].upper()[2:] for i in args]
		values = [i.split('=')[1] for i in args]
		n_procs = int(values[keys.index('PROCESS')]) + 1

		if n_procs in range(1, cpu_count() + 1):

			# ensure reproducibility
			set_determinism(seed=3)
			random.seed(3)

			# load and splitting data
			data_folder = make_dataset(dataset='glioma', verbose=False, base_path=_base_path)
			train_data, eval_data, test_data = train_test_splitting(data_folder, reports_path=reports_path, load_from_file=True, verbose=False)
			l = len(train_data) + len(eval_data) + len(test_data)
			data = np.concatenate([train_data, eval_data, test_data]).reshape(l)

			# defining model
			model = AutoEncoder3D(
				spatial_dims = 3,
				in_channels = 4,
				out_channels = 4,
				channels = (5,),
				strides = (2,),
				inter_channels = (8, 8, 16),
				inter_dilations = (1, 2, 4)
			)
			# model = None ## NOTE: uncomment to compute only intensity features.

			intervals = np.linspace(0, l, n_procs).astype(int)
			for i in range(len(intervals) - 1):
				procs = []
				proc = Process(
					target=target_function,
					args=(data[intervals[i]:intervals[i+1]], model, random.randint(1000, 9999),)
				)
				procs.append(proc)
				proc.start()

			# complete the processes
			for proc in procs:
				proc.join()
		else:
			print('\n' + ''.join(['> ' for i in range(25)]))
			print('\nWARNING: out-of-bound parameters!')
			print('\n' + ''.join(['> ' for i in range(25)]))
			print(f'\n{"PARAM":<16}{"VALUE RANGE":<18}\n')
			print(''.join(['> ' for i in range(25)]))
			print(f'{"--process":<16}{("min: 1 and max: " + str(cpu_count()) + "."):<18}')
			print(''.join(['> ' for i in range(25)]) + '\n')

