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
os.environ['OPENBLAS_NUM_THREADS'] = '7'
os.environ['MKL_NUM_THREADS'] = '7'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



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
	return tf.stack(list(features.values()), axis=-1), tf.one_hot(labels, depth=3)

def pack_b(features, labels):
	return tf.stack(list(features.values()), axis=-1), tf.one_hot(labels, depth=4)


if __name__ == '__main__':
	usage = '%s [-opt1, [-opt2, ...]] directory' % __file__
	parser = argparse.ArgumentParser(description='', formatter_class=RawTextHelpFormatter, usage=usage)
	parser.add_argument('directory', type=is_valid_file, help='input directory')
	parser.add_argument('-o', '--outfile', action="store", default=sys.stdout, type=argparse.FileType('w'), help='where to write output [stdout]')
	parser.add_argument('-c', '--columns', action="store", default="GC", help='sel cols')
	parser.add_argument('-b', '--both', action="store_true")
	parser.add_argument('-d', '--deep', action="store_true")
	args = parser.parse_args()

	cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=args.directory + "_" + args.columns + '_deep.ckpt',save_weights_only=True,verbose=1)

	stops = ['#', '*', '+']
	letters = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
	codings = [ str(a) for a in range(8) ]
	dicodings = [ str(a)+':'+str(b) for a in range(8) for b in range(8)]
	tricodings = [ str(a)+':'+str(b)+':'+str(c) for a in range(8) for b in range(8) for c in range(8) ]
	dimers = [a+':'+b for a in stops+letters for b in stops+letters]
	dipeps = [a+'_'+b for a in stops+letters for b in stops+letters]
	trimers = [a+':'+b+':'+c for a in stops+letters for b in stops+letters for c in stops+letters]
	######  single  ######
	colnames =  ['TYPE', 'GC', 'GCw','a','c','g','t','GC1','GC2','GC3', '1a','1c','1g','1t', '2a','2c','2g','2t', '3a','3c','3g','3t'] + stops + letters + codings + dicodings + tricodings + dimers + dipeps
	if 'TRICODINGS' in args.columns:
		args.columns = args.columns.replace('TRICODINGS', ','.join(tricodings) )
	if 'DICODINGS' in args.columns:
		args.columns = args.columns.replace('DICODINGS', ','.join(dicodings) )
	if 'CODINGS' in args.columns:
		args.columns = args.columns.replace('CODINGS', ','.join(codings) )
	if 'DIMERS' in args.columns:
		args.columns = args.columns.replace('DIMERS', ','.join(dimers) )
	if 'DIPEPS' in args.columns:
		args.columns = args.columns.replace('DIPEPS', ','.join(dipeps) )
	if 'TRIMERS' in args.columns:
		args.columns = args.columns.replace('TRIMERS', ','.join(trimers) )
		colnames =  ['TYPE', 'GC', '1a','1c','1g','1t', '2a','2c','2g','2t', '3a','3c','3g','3t'] + stops + letters + trimers
	selnames = ['TYPE'] + args.columns.split(',') + letters 

	
	######   glob   ######
	#colnames =  ['TYPE', 'GC', 'a','t','g', 'c'] + [letter+f for f in ['+0','-0','+1','-1','+2','-2'] for letter in letters]
	#colnames =  ['TYPE', 'GC', 'a','t','g', 'c', 'P1', 'P2', 'P3'] + [letter+f for f in ['+0','-0','+1','-1','+2','-2'] for letter in letters]

	# BOTH
	if args.both:
		colnames = ['TYPE', 'GC'] + [f+d for d in ['+','-'] for f in colnames[2:] ]
		selnames = ['TYPE', 'GC'] + [f+d for d in ['+','-'] for f in selnames[2:] ]
		model = mm.create_model_b(len(selnames)-1)
	elif args.deep:
		model = mm.create_model_d(len(selnames)-1)
	else:
		model = mm.create_model(len(selnames)-1)
	
	##############################################
	print("Using", len(selnames), "features")
	print(colnames)
	print(selnames)

	#exit()	
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

	if args.both:
		tdata = tfiles.map(pack_b)
	else:
		tdata = tfiles.map(pack)
	#tdata = tfiles.map( lambda features,labels: tf.stack(list(features.values()), axis=-1), tf.one_hot(labels, depth=3) )
	

	#for feature in tfiles.take(1):
	#	print( len(np.unique(feature[0]['GC'].numpy())) )
	#exit()
	#metrics = Metrics()
	#exit()
	vfiles = tf.data.experimental.make_csv_dataset(
		compression_type    = 'GZIP',
		#file_pattern        = args.directory + "/NC_?????[1357]*.tsv",
		#file_pattern        = args.directory + "/NC_0[01]*.tsv",
		file_pattern        = args.directory + "/NC_02*.tsv.gz",
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
	if args.both:
		vdata = vfiles.map(pack_b)
	else:
		vdata = vfiles.map(pack)
	
	
	#class_weight = {0:1, 1:1, 2:1}
	with tf.device('/device:CPU:0'):
		#es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, min_delta=0.001, baseline=None)
		#lr_callback = LRFinder()
		model.fit(tdata,
				  validation_data = vdata,
				  epochs          = 100,
				  #class_weight   = class_weight,
				  verbose         = 0,
				  callbacks       = [LossHistoryCallback(), cp_callback]
		)

	#lr_callback.plot()

