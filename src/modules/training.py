"""
Functions for training the models.
"""
import os, random, time, calendar, datetime
import pandas as pd
import numpy as np
import torch
from monai.data import DataLoader, CacheDataset
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from torch_geometric.loader import DataLoader as DataLoaderGraph
from src.helpers.utils import get_date_time, save_results, get_class_weights, get_brats_classes
from src.modules.metrics import (
	calculate_ssim,
	calculate_psnr,
	calculate_ncc,
	calculate_node_dice
)


__all__ = ['train_test_splitting', 'training_model', 'predict_ae']


def train_test_splitting(
		folder,
		train_ratio = .8,
		reports_path = None,
		write_to_file = False,
		load_from_file = False,
		verbose = True
	):
	"""
	Splitting train/eval/test.
	Args:
		folder (str): the path of the folder containing data.
		train_ratio (float): ratio of the training set, value between 0 and 1.
		reports_path (str): folder where to save report file.
			Required if `write_to_file` is True and/or `load_from_file` is True.
		write_to_file (bool): whether to write selected data to csv file.
		load_from_file (bool): whether to load splitting data from a previous saved csv file.
		verbose (bool): whether or not print information.
	Returns:
		train_data (list): the training data ready to feed monai.data.Dataset.
		eval_data (list): the evaluation data ready to feed monai.data.Dataset.
		test_data (list): the testing data ready to feed monai.data.Dataset.
		(See https://docs.monai.io/en/latest/data.html#monai.data.Dataset).
	"""
	sessions = ['-'.join(s.split('-')[:3]) for s in os.listdir(folder) if os.path.isdir(os.path.join(folder, s))]
	subjects = list(set(sessions))
	random.shuffle(subjects)
	split_train = int(len(subjects) * train_ratio)
	train_subjects, test_subjects = subjects[:split_train], subjects[split_train:]
	split_eval = int(len(train_subjects) * .8)
	eval_subjects = train_subjects[split_eval:]
	train_subjects = train_subjects[:split_eval]
	if load_from_file:
		if not reports_path:
			print('\n' + ''.join(['> ' for i in range(30)]))
			print('\nERROR: Paremeter \033[95m `reports_path`\033[0m must be specified.\n')
			print(''.join(['> ' for i in range(30)]) + '\n')
			return [],[],[]
		else:
			try:
				df_split = pd.read_csv(os.path.join(reports_path, [i for i in os.listdir(reports_path) if 'splitting' in i][0]))
			except Exception:
				print('\n' + ''.join(['> ' for i in range(40)]))
				print('\nERROR: Paremeter \033[95m `load_from_file`\033[0m set to \033[95m `True`\033[0m but no splitting file was found.\n')
				print(''.join(['> ' for i in range(40)]) + '\n')
			train_subjects = df_split['train_subjects'].dropna().to_numpy()
			eval_subjects = df_split['eval_subjects'].dropna().to_numpy()
			test_subjects = df_split['test_subjects'].dropna().to_numpy()
	if write_to_file:
		if not reports_path:
			print('\n' + ''.join(['> ' for i in range(30)]))
			print('\nERROR: Paremeter \033[95m `reports_path`\033[0m must be specified.\n')
			print(''.join(['> ' for i in range(30)]) + '\n')
			return [],[],[]
		else:
			for i in range(max(len(train_subjects), len(eval_subjects), len(test_subjects))):
				save_results(
					os.path.join(reports_path, 'splitting_'+str(calendar.timegm(time.gmtime()))+'.csv'),
					{
						'train_subjects': train_subjects[i] if i < len(train_subjects) else '',
						'eval_subjects': eval_subjects[i] if i < len(eval_subjects) else '',
						'test_subjects': test_subjects[i] if i < len(test_subjects) else ''
					}
				)
	train_sessions = [os.path.join(folder, s) for s in os.listdir(folder) if '-'.join(s.split('-')[:3]) in train_subjects]
	eval_sessions = [os.path.join(folder, s) for s in os.listdir(folder) if '-'.join(s.split('-')[:3]) in eval_subjects]
	test_sessions = [os.path.join(folder, s) for s in os.listdir(folder) if '-'.join(s.split('-')[:3]) in test_subjects]
	train_labels = [os.path.join(s, s.split('/')[-1] + '-seg.nii.gz') for s in train_sessions]
	eval_labels = [os.path.join(s, s.split('/')[-1] + '-seg.nii.gz') for s in eval_sessions]
	test_labels = [os.path.join(s, s.split('/')[-1] + '-seg.nii.gz') for s in test_sessions]
	modes = ['t1c', 't1n', 't2f', 't2w']
	train_data, eval_data, test_data = {}, {}, {}
	train_data = [dict({
		'image': [os.path.join(s, s.split('/')[-1] + '-' + m + '.nii.gz') for m in modes],
		'label': train_labels[i],
		'subject': s.split('/')[-1]
	}) for i, s in enumerate(train_sessions)]
	eval_data = [dict({
		'image': [os.path.join(s, s.split('/')[-1] + '-' + m + '.nii.gz') for m in modes],
		'label': eval_labels[i],
		'subject': s.split('/')[-1]
	}) for i, s in enumerate(eval_sessions)]
	test_data = [dict({
		'image': [os.path.join(s, s.split('/')[-1] + '-' + m + '.nii.gz') for m in modes],
		'label': test_labels[i],
		'subject': s.split('/')[-1]
	}) for i, s in enumerate(test_sessions)]
	if verbose:
		print(''.join(['> ' for i in range(40)]))
		print(f'\n{"":<20}{"TRAINING":<20}{"EVALUATION":<20}{"TESTING":<20}\n')
		print(''.join(['> ' for i in range(40)]))
		tsb1 = str(len(train_subjects)) + ' (' + str(round((len(train_subjects) * 100 / len(subjects)), 0)) + ' %)'
		tsb2 = str(len(eval_subjects)) + ' (' + str(round((len(eval_subjects) * 100 / len(subjects)), 0)) + ' %)'
		tsb3 = str(len(test_subjects)) + ' (' + str(round((len(test_subjects) * 100 / len(subjects)), 0)) + ' %)'
		tss1 = str(len(train_sessions)) + ' (' + str(round((len(train_sessions) * 100 / len(sessions)), 2)) + ' %)'
		tss2 = str(len(eval_sessions)) + ' (' + str(round((len(eval_sessions) * 100 / len(sessions)), 2)) + ' %)'
		tss3 = str(len(test_sessions)) + ' (' + str(round((len(test_sessions) * 100 / len(sessions)), 2)) + ' %)'
		print(f'\n{"subjects":<20}{tsb1:<20}{tsb2:<20}{tsb3:<20}\n')
		print(f'{"sessions":<20}{tss1:<20}{tss2:<20}{tss3:<20}\n')
	return train_data, eval_data, test_data


def training_model(
		model,
		data,
		transforms,
		epochs,
		device,
		paths,
		val_interval = 1,
		early_stopping = 10,
		num_workers = 4,
		ministep = 12,
		batch_size = 1,
		lr = 1e-4,
		weight_decay = 1e-5,
		run_id = '',
		write_to_file = True,
		verbose = False
	):
	"""
	Standard Pytorch-style training program.
	Args:
		model (torch.nn.Module): the model to be trained.
		data (list): the training and evalutaion data.
		transform (list): transformation sequence for training and evaluation data.
		epochs (int): max number of epochs.
		device (str): device's name.
		paths (list): folders where to save results and model's dump.
		val_interval (int): validation interval.
		early_stopping (int): nr. of epochs for those there's no more improvements.
		num_workers (int): setting multi-process data loading.
		ministep (int): number of interval of data to load on RAM.
		batch_size (int): the size of the batch.
		lr (float): learning rate for the optimizer.
		weight_decay (float): weight decay for the optimizer.
		run_id (str): whether to set a specific training identifier or using a default.
		write_to_file (bool): whether to write results to csv file.
		verbose (bool): whether to print minimal or extended information.
	Returns:
		metrics (list): the list of all the computed metrics over the training in this order:
			- losses during training;
			- losses during evaluation;
			- execution times by epochs;
			- metrics during evaluation (based on input `model`);
	"""
	# unfolds grouped data/init model and utils
	device = torch.device(device)
	model = model.to(device)
	train_data, eval_data = data
	train_transform, eval_transform = transforms
	saved_path, reports_path, logs_path, graph_path = paths
	ministep = ministep if (len(train_data) > 10 and len(eval_data) > 10 and ministep > 1) else 2

	# define loss, optimizer, scheduler
	if model.name == 'AutoEncoder3D':
		loss_function = torch.nn.MSELoss()
	else:
		class_weights = get_class_weights((train_data + eval_data), graph_path)
		loss_function = torch.nn.CrossEntropyLoss(weight=class_weights.to(device), reduction='mean')
	optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
	lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
	scaler = torch.GradScaler(device=str(device)) # use Automatic Mixed Precision to accelerate training
	torch.backends.cudnn.benchmark = True # enable cuDNN benchmark

	# define metric/loss collectors
	best_metric, best_metric_epoch = -1, -1
	best_metrics_epochs_and_time = [[], [], []]
	epoch_loss_values, epoch_time_values, epoch_metric_values = [[], []], [], [[], [], []]

	# log the current execution
	run_id = run_id if run_id else model.name.upper() + '_' + str(calendar.timegm(time.gmtime()))
	log = open(os.path.join(logs_path, 'training.log'), 'a', encoding='utf-8')
	log.write('['+get_date_time()+'] Training phase started.EXECUTING: ' + run_id + '\n')
	log.flush()
	total_start = time.time()
	for epoch in range(epochs):
		epoch_start = time.time()
		print(''.join(['> ' for i in range(40)]))
		print(f"epoch {epoch + 1}/{epochs} run_id {run_id}")
		log.write('['+get_date_time() + '] EXECUTING.' + run_id + ' EPOCH ' + str(epoch + 1) + ' OF ' + str(epochs) + ' \n')
		log.flush()
		model.train()
		epoch_loss_train, epoch_loss_eval = 0, 0
		metric_ssim, metric_psnr, metric_ncc, metric_dice = 0, 0, 0, 0
		step_train, step_eval = 0, 0
		ministeps_train = np.linspace(0, len(train_data), ministep).astype(int)
		ministeps_eval = np.linspace(0, len(eval_data), ministep).astype(int)

		# start training
		for i in range(len(ministeps_train) - 1):
			if model.name == 'AutoEncoder3D':
				train_ds = CacheDataset(train_data[ministeps_train[i]:ministeps_train[i+1]], transform=train_transform, cache_rate=1.0, num_workers=None, progress=False)
				train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
			else:
				data_list = [train_transform(d)['graph'] for d in train_data[ministeps_train[i]:ministeps_train[i+1]]]
				train_loader = DataLoaderGraph(data_list, batch_size=batch_size, shuffle=True)
			for batch_data in train_loader:
				step_start = time.time()
				step_train += 1
				input = batch_data['image'].to(device) if model.name == 'AutoEncoder3D' else batch_data.to(device)
				optimizer.zero_grad()
				with torch.autocast(device_type=str(device)):
					if model.name == 'AutoEncoder3D':
						_, reconstruction = model(input)
						loss = loss_function(reconstruction, input)
					else:
						out = model(input.x, input.edge_index.type(torch.int64))
						loss = loss_function(out, input.y.long())

				scaler.scale(loss).backward()
				scaler.step(optimizer)
				scaler.update()
				epoch_loss_train += loss.item()
				if verbose:
					print(
						f"{step_train}/{len(train_data) // train_loader.batch_size}"
						f", train_loss: {loss.item():.4f}"
						f", step time: {str(datetime.timedelta(seconds=int(time.time() - step_start)))}"
					)
		lr_scheduler.step()
		epoch_loss_train /= step_train
		epoch_loss_values[0].append(epoch_loss_train)
		print(f"epoch {epoch + 1} average training loss: {epoch_loss_train:.4f}")

		# start validation
		if (epoch + 1) % val_interval == 0:
			model.eval()
			with torch.no_grad():
				for i in range(len(ministeps_eval) - 1):
					if model.name == 'AutoEncoder3D':
						eval_ds = CacheDataset(eval_data[ministeps_eval[i]:ministeps_eval[i+1]], transform=eval_transform, cache_rate=1.0, num_workers=None, progress=False)
						eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
					else:
						data_list = [eval_transform(d)['graph'] for d in eval_data[ministeps_eval[i]:ministeps_eval[i+1]]]
						eval_loader = DataLoaderGraph(data_list, batch_size=batch_size, shuffle=True)
					for val_data in eval_loader:
						step_eval += 1
						if model.name == 'AutoEncoder3D':
							val_input = val_data['image'].to(device)
							_, val_reconstruction = model(val_input)
							val_loss = loss_function(val_reconstruction, val_input)
							metric_ssim += calculate_ssim(val_reconstruction, val_input)
							metric_psnr += calculate_psnr(val_reconstruction, val_input)
							metric_ncc += calculate_ncc(val_reconstruction, val_input)
						else:
							val_input = val_data.to(device)
							out = model(val_input.x, val_input.edge_index.type(torch.int64))
							val_loss = loss_function(out, val_input.y.long())
							pred = out.argmax(dim=1)
							dice = calculate_node_dice(pred.float(), val_input.y.float())
							metric_dice += dice.mean()

						epoch_loss_eval += val_loss.item()

				epoch_loss_eval /= step_eval
				epoch_loss_values[1].append(epoch_loss_eval)

				# calculate metrics
				if model.name == 'AutoEncoder3D':
					metric_ssim /= step_eval
					epoch_metric_values[0].append(metric_ssim)
					metric_psnr /= step_eval
					epoch_metric_values[1].append(metric_psnr)
					metric_ncc /= step_eval
					epoch_metric_values[2].append(metric_ncc)
				else:
					metric_dice /= step_eval
					epoch_metric_values[0].append(metric_dice)

				# save best performing model
				metric_to_check = metric_ssim if model.name == 'AutoEncoder3D' else metric_dice
				if metric_to_check > best_metric:
					best_metric = metric_to_check
					best_metric_epoch = epoch + 1
					best_metrics_epochs_and_time[0].append(best_metric)
					best_metrics_epochs_and_time[1].append(best_metric_epoch)
					best_metrics_epochs_and_time[2].append(time.time() - total_start)
					torch.save(model.state_dict(), os.path.join(saved_path, run_id + '_best.pth'))
					print("saved new best model")
				print(
					f"current epoch: {epoch + 1} current mean score: {metric_to_check:.4f}"
					f"\nbest mean score: {best_metric:.4f}"
					f" at epoch: {best_metric_epoch}"
				)
		print(f"time consuming of epoch {epoch + 1} is: {str(datetime.timedelta(seconds=int(time.time() - epoch_start)))}")
		epoch_time_values.append(time.time() - epoch_start)

		# save results to file
		if write_to_file:
			metrics = {
					'id': run_id,
					'epoch': epoch + 1,
					'model': model.name,
					'train_loss': epoch_loss_train,
					'eval_loss': epoch_loss_eval,
					'exec_time': time.time() - epoch_start
			}
			if model.name == 'AutoEncoder3D':
				metrics['metric_ssim'] = metric_ssim
				metrics['metric_psnr'] = metric_psnr
				metrics['metric_ncc'] = metric_ncc
			else:
				metrics['metric_dice_avg'] = metric_dice.cpu().numpy()
			metrics['datetime'] = get_date_time()
			save_results(
				file = os.path.join(reports_path, model.name.upper() + '_training.csv'),
				metrics = metrics
			)

		# early stopping
		if epoch + 1 - best_metric_epoch == early_stopping:
			print(f"\nEarly stopping triggered at epoch: {str(epoch + 1)}\n")
			break

	print(f"\n\nTrain completed! Best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, total time: {str(datetime.timedelta(seconds=int(time.time() - total_start)))}.")
	log.write('['+get_date_time()+'] Training phase ended.EXECUTING: ' + run_id + '\n')
	log.flush()
	log.close()
	return [
		epoch_loss_values[0],
		epoch_loss_values[1],
		epoch_time_values,
		epoch_metric_values
	]


def predict_ae(
	model,
	data,
	transforms,
	device,
	saved_path,
	saved_model = ''
):
	"""
	Standard Pytorch-style prediction program. Accepts a single data example, not batch.
	Args:
		model (torch.nn.Module): the model to be loaded.
		data (dict): a single testing data example returned from Dataloader class.
		transforms (list): data transformations pipeline.
		device (str): device's name.
		saved_path (list): folders where to load model's dump.
		saved_model (str): name of a model's dump to be loaded. If empty string is passed the latest
			in cronological order will be taken.
	Returns:
		test_reconstruction (monai.data.meta_tensor.MetaTensor): the AE reconstructed image.
		latent (monai.data.meta_tensor.MetaTensor): the AE latent feature maps.
			Expected shape to be (16, 112, 112, 72) if input shape (224, 224, 144).
		metrics (list): list of metrics computed in this order [SSIM, PSNR, NCC].
	"""
	# unfolds grouped data/init model and utils
	device = torch.device(device)
	model = model.to(device)
	try:
		# load pretrained model
		last_model = saved_model if saved_model else sorted([n for n in os.listdir(saved_path) if model.name.upper() in n and '.pth' in n])[-1]
		model.load_state_dict(
			torch.load(os.path.join(saved_path, last_model), map_location=torch.device(device))
		)
		model.eval()
		# making inference
		with torch.no_grad():
			test_input = transforms(data)
			test_input = test_input['image'].to(device)
			latent, test_reconstruction = model(test_input)
			metrics = [
				calculate_ssim(test_reconstruction, test_input),
				calculate_psnr(test_reconstruction, test_input),
				calculate_ncc(test_reconstruction, test_input)
			]
			return test_reconstruction, latent, metrics
	except OSError as e:
		print('\n' + ''.join(['> ' for i in range(30)]))
		print('\nERROR: model dump for\033[95m '+model.name+'\033[0m not found.\n')
		print(''.join(['> ' for i in range(30)]) + '\n')


def predict_gnn(
	model,
	data,
	transforms,
	device,
	paths,
	saved_model = '',
	write_to_file = True,
	return_predictions = True,
	verbose = False
):
	"""
	Standard Pytorch-style prediction program.
	Args:
		model (torch.nn.Module): the model to be loaded.
		data (list): the testing data.
		transform (list): pre and post transformations for testing data.
		device (str): device's name.
		paths (list): folders where to save results and load model's dump.
		saved_model (str): name of a model's dump to be loaded. If empty string is passed the latest
			in cronological order will be taken.
		write_to_file (bool): whether to write results to csv file.
		return_predictions (bool): whether to return the predicted masks.
		verbose (bool): whether or not print information.
	Returns:
		metrics (list): node dice score, dice score and Hausdorff distance for each class.
		predictions (list): the predicted mask if `return_predictions` is set to `True`.
	"""
	# unfolds grouped data/init model and utils
	device = torch.device(device)
	model = model.to(device)
	l, preds, node_dice = len(data), [], 0
	saved_path, reports_path, logs_path = paths

	# define metrics
	dice_metric_batch = DiceMetric(include_background=True, reduction='mean_batch')
	hausdorff_metric_batch = HausdorffDistanceMetric(include_background=True, reduction='mean_batch', percentile=95)

	# log the current execution
	log = open(os.path.join(logs_path, 'prediction.log'), 'a', encoding='utf-8')
	log.write('['+get_date_time()+'] Predictions started.EXECUTING: ' + model.name + '\n')
	log.flush()

	try:
		# load pretrained model
		last_model = saved_model if saved_model else sorted([n for n in os.listdir(saved_path) if model.name.upper() in n and '.pth' in n])[-1]
		model.load_state_dict(
			torch.load(os.path.join(saved_path, last_model), map_location=torch.device(device))
		)
		model.eval()
		# making inference
		with torch.no_grad():
			for k, d in enumerate(data):
				data_adj = transforms(d)
				test_input = data_adj['graph'].to(device)
				out = model(test_input.x, test_input.edge_index.type(torch.int64))
				pred = out.argmax(dim = 1)
				pred_adj = pred.type(torch.int64).cpu().numpy()

				# computing the prediction masks
				if return_predictions:
					pred_mask = np.zeros(data_adj['label'][0].shape)
					for lab in range(1, 4):
						op_set = set(np.flatnonzero(pred_adj == lab))
						for j, points in enumerate(data_adj['map']):
							if j in op_set:
								pred_mask[tuple(np.array(points).T)] = lab
					pred_mask = get_brats_classes(pred_mask)
					label_mask = get_brats_classes(data_adj['label'][0])
					preds.append(pred_mask)

				# compute metrics
				node_dice += calculate_node_dice(pred.float(), test_input.y.float()).mean()
				if return_predictions:
					dice_metric_batch(y_pred=torch.tensor([pred_mask]), y=[label_mask])
					hausdorff_metric_batch(y_pred=torch.tensor([pred_mask]), y=[label_mask])
				if verbose and (k == 0 or (k % int(l / (10 if l >= 10 else 1))) == 0):
					print(f"inference {k}/{l}")
					log.write('['+get_date_time()+'] EXECUTING.'+model.name+' INFERENCE '+str(k)+' OF '+str(l)+' \n')
					log.flush()
			if return_predictions:
				dice_batch_org = dice_metric_batch.aggregate()
				hausdorff_batch_org = hausdorff_metric_batch.aggregate()
				dice_metric_batch.reset()
				hausdorff_metric_batch.reset()
				metrics = [
					[i.item() for i in dice_batch_org],
					[j.item() for j in hausdorff_batch_org]
				]
			else:
				metrics = [[],[]]
			node_dice /= l
			metrics.append([np.float32(node_dice.cpu().numpy())])

			dump = {
				'model': model.name,
				'n_images': l,
				'node_dice_score': metrics[2][0],
			}
			if return_predictions:
				dump['dice_score_et'] = metrics[0][0]
				dump['dice_score_tc'] = metrics[0][1]
				dump['dice_score_wt'] = metrics[0][2]
				dump['hausdorff_score_et'] = metrics[1][0]
				dump['hausdorff_score_tc'] = metrics[1][1]
				dump['hausdorff_score_wt'] = metrics[1][2]
			dump['datetime'] = get_date_time()

			# save results to file
			if write_to_file:
				save_results(
					file = os.path.join(reports_path, model.name.upper() + '_testing.csv'),
					metrics = dump
				)
			log.write('['+get_date_time()+'] Predictions ended.EXECUTING: ' + model.name + '\n')
			log.flush()
			log.close()
			return metrics, preds
	except OSError as e:
		print('\n' + ''.join(['> ' for i in range(30)]))
		print('\nERROR: model dump for\033[95m '+model.name+'\033[0m not found.\n')
		print(''.join(['> ' for i in range(30)]) + '\n')
