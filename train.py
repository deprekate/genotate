import os
import sys
import re
import argparse
from argparse import RawTextHelpFormatter
from statistics import mode

import genotate.make_model as mm

# TensorFlow and tf.keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

#sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

# Helper libraries
os.environ['OPENBLAS_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt



class LossHistoryCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, batch, logs=None):
		#logs['loss'] or logs['val_loss'] (the latter available only if you use validation data when model.fit()
		# Use logs['loss'] or logs['val_loss'] for pyqt5 purposes
		row = list()
		for key, value in logs.items():
			row.append(key)
			row.append(value)
		print('\t'.join(map(str,row)), flush=True)

class LearningRateReducerCb(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		old_lr = self.model.optimizer.lr.read_value()
		new_lr = old_lr * 0.99
		print("\nEpoch: {}. Reducing Learning Rate from {} to {}".format(epoch, old_lr, new_lr))
		self.model.optimizer.lr.assign(new_lr)


def is_valid_file(x):
	if not os.path.exists(x):
		raise argparse.ArgumentTypeError("{0} does not exist".format(x))
	return x

def pack(features, labels):
	#return tf.stack(list(features.values()), axis=-1), labels
	#return tf.stack(list(features.values()), axis=-1), tf.one_hot(tf.add(labels,2), depth=4)
	return tf.stack(list(features.values()), axis=-1), tf.one_hot(labels, depth=4)



if __name__ == '__main__':
	usage = '%s [-opt1, [-opt2, ...]] directory' % __file__
	parser = argparse.ArgumentParser(description='', formatter_class=RawTextHelpFormatter, usage=usage)
	parser.add_argument('directory', type=is_valid_file, help='input directory')
	parser.add_argument('-o', '--outfile', action="store", default=sys.stdout, type=argparse.FileType('w'), help='where to write output [stdout]')
	parser.add_argument('-c', '--columns', action="store", default="GC,a,c,g,t", help='sel cols')
	args = parser.parse_args()

	stops = ['#', '*', '+']
	letters = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
	######  single  ######
	#colnames = ['TYPE', 'GC', 'a','c','g', 't'] + letters
	#colnames = ['TYPE', 'GC', 'a','c','g', 't'] + letters + [str(a)+':'+str(b) for a in range(7) for b in range(7)]
	#colnames =  ['TYPE', 'GC', 'a','t','g', 'c', 'P1', 'P2', 'P3'] + letters
	#colnames =  ['TYPE', 'GC', 'GCw','a','c','g','t','GC1','GC2','GC3', '1a','1c','1g','1t', '2a','2c','2g','2t', '3a','3c','3g','3t'] + stops + letters + [str(a)+':'+str(b) for a in range(7) for b in range(7)]
	colnames =  ['TYPE', 'GC', '1a','1c','1g','1t', '2a','2c','2g','2t', '3a','3c','3g','3t'] + stops + letters + [a+':'+b for a in stops+letters for b in stops+letters]
	#colnames =  ['TYPE', 'GC', 'GCw','a','c','g','t','GC1','GC2','GC3', '1a','1c','1g','1t', '2a','2c','2g','2t', '3a','3c','3g','3t'] + stops + letters + [a+':'+b for a in stops+letters for b in stops+letters]
	######   glob   ######
	#colnames =  ['TYPE', 'GC', 'a','t','g', 'c'] + [letter+f for f in ['+0','-0','+1','-1','+2','-2'] for letter in letters]
	#colnames =  ['TYPE', 'GC', 'a','t','g', 'c', 'P1', 'P2', 'P3'] + [letter+f for f in ['+0','-0','+1','-1','+2','-2'] for letter in letters]
	selnames = ['TYPE'] + args.columns.split(',') + letters 

	selnames = colnames
	
	# BOTH
	'''
	colnames = ['TYPE', 'GC'] + [f+d for d in ['+','-'] for f in colnames[2:] ]
	sel =  ['TYPE', 'GC', '1a','1c','1g','1t', '2a','2c','2g','2t', '3a','3c','3g','3t'] + stops + letters + [a+':'+b for a in stops+letters for b in stops+letters]
	selnames = ['TYPE', 'GC'] + [f+d for d in ['+','-'] for f in sel[2:] ]
	'''

	#selnames = ['TYPE', 'GC'] + [f+d for d in ['+','-'] for f in args.columns.replace('GC,','').split(',') + letters ]

	#colnames = ['TYPE', 'GC', 'a','t','g','c'] + [letter for pair in zip([l+'1' for l in letters], [l+'2' for l in letters]) for letter in pair]
	#colnames = ['TYPE', 'GC', 'a1','t1','g1', 'c1'] + [c + '1' for c in letters] + ['a2','t2','g2', 'c2'] + [c + '2' for c in letters]
	#colnames =  ['TYPE', 'GC', 'a','t','g', 'c'] + [letter+d+f for f in ['+0','-0','+1','-1','+2','-2'] for letter in letters for d in 'ab']
	#selnames = colnames[:2] + colnames[2:]

	tfiles = tf.data.experimental.make_csv_dataset(
		compression_type    = 'GZIP',
		file_pattern        = args.directory + "/NC_0[01]*.tsv.gz",
		#file_pattern        = args.directory + "/NC_001416.tsv",
		field_delim         = '\t',
		header              = False,
		column_names        = colnames,
		select_columns      = selnames,
		column_defaults     = [tf.int32] + [tf.float32] * (len(selnames)-1),
		shuffle             = True,
		num_parallel_reads  = 1000,
		shuffle_buffer_size = 5000,
		batch_size          = 500,
		num_epochs          = 1,
		sloppy				= True,
		label_name          = colnames[0]
		)
	tdata = tfiles.map(pack)
	#tdata = tfiles.map( lambda features,labels: tf.stack(list(features.values()), axis=-1), tf.one_hot(labels, depth=3) )
	
	model = mm.create_model(len(selnames)-1)

	#for feature in tfiles.take(1):
	#	print( len(np.unique(feature[0]['GC'].numpy())) )
	#exit()
	#metrics = Metrics()
	#exit()
	vfiles = tf.data.experimental.make_csv_dataset(
		compression_type    = 'GZIP',
		#file_pattern        = args.directory + "/NC_?????[1357]*.tsv",
		#file_pattern        = args.directory + "/NC_0[01]*.tsv",
		file_pattern        = args.directory + "/NC_0*.tsv.gz",
		#file_pattern        = "data/train/single/NC_001416.tsv.gz",
		field_delim         = '\t',
		header              = False,
		column_names        = colnames,
		select_columns      = selnames,
		column_defaults     = [tf.int32] + [tf.float32] * (len(selnames)-1),
		batch_size          = 100,
		num_epochs          = 1,
		shuffle             = True,
		num_parallel_reads  = 10,
		sloppy				= True,
		label_name          = colnames[0]
		#label_name          = 0 # this is old version
		)
	vdata = vfiles.map(pack)
	#Xval = pd.read_csv(args.directory + "/NC_001416.tsv",names=colnames, sep='\t')
	#Yval = Xval.pop('TYPE')
	
	
	model = mm.create_model(len(selnames)-1)

	#class_weight = {0:1, 1:1, 2:1}
	with tf.device('/device:GPU:0'):
		#cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=args.directory + '_' + re.sub('0.*6', 'dicodings', args.columns) + '.ckpt',save_weights_only=True,verbose=1)
		cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=args.directory + '_kmers.ckpt',save_weights_only=True,verbose=1)
		#cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=args.directory + '_self.ckpt',save_weights_only=True,verbose=1)
		#es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, min_delta=0.001, baseline=None)
		#lr_callback = LRFinder()
		model.fit(tdata,
				  validation_data = vdata,
				  epochs          = 100,
				  #class_weight   = class_weight,
				  verbose         = 1,
				  callbacks       = [LossHistoryCallback(), cp_callback]
		)

	#lr_callback.plot()

