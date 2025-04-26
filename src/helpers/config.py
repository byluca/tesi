"""
The collection of global configurations
"""
CONFIG = {
	'DATA_FOLDER':			'data/',
	'SAVED_FOLDER': 		'saved/',
	'REPORT_FOLDER':		'reports/',
	'LOG_FOLDER':			'logs/',
	'SYN_IDS':				{
		'glioma':		'syn51514105',
		'africa':		'syn51514109',
		'meningioma':	'syn51514106',
		'metastasis':	'syn51514107',
		'pediatric':	'syn51514108'
	},
	'CHANNELS':			[
		'post-contrast T1-weighted',
		'T1-native',
		'T2-FLAIR',
		'T2-weighted',
		'segmentation mask'
	],
	'LABELS': 			[
		'NCR (necrotic tumor core)',
		'ED (peritumoral edematous/invaded tissue)',
		'ET (GD-enhancing tumor)'
	],
	'CLASSES':			[
		'ET (Enhancing Tumor)',
		'TC (Tumor Core)',
		'WT (Whole Tumor)'
	],
	'GRAPH_FOLDER':		'graphs/',
	'GNN':				['GraphSAGE', 'GAT', 'ChebNet'],
	'HP_TUNING':		{
		'SHARED':	{
			'lr': [1e-4, 1e-5],
			'weight_decay': [1e-5, 1e-6],
			'dropout': [.0, .1, .5],
			'hidden_channels': [
				[256, 256, 256, 256],
				[512, 512, 512, 512],
				[256, 256, 256, 256, 256, 256, 256],
				[512, 512, 512, 512, 512, 512, 512],
			]
		},
		'GraphSAGE':	{
			'aggr': ['mean', 'max']
		},
		'GAT':			{
			'heads': [6, 14],
			'attention_dropout': [.0, .1, .5]
		},
		'ChebNet':		{
			'k': [2, 3, 4]
		}
	}
}


def get_config():
	"""
	The getter method
	"""
	return CONFIG


def set_config(key, value):
	"""
	The setter method
	"""
	CONFIG.update({key: value})