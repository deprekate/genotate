import os
import sys
import re
import argparse
from argparse import RawTextHelpFormatter

import genotate.make_model as mm

import resource
low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

# TensorFlow and tf.keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
#sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
	tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Helper libraries
os.environ['OPENBLAS_NUM_THREADS'] = '9'
os.environ['MKL_NUM_THREADS'] = '9'
#import matplotlib.pyplot as plt

def compute_loss():
	y_pred_model_w_temp = tf.math.divide(y_pred, temp)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf.convert_to_tensor(tf.keras.utils.to_categorical(Y)), y_pred_model_w_temp))
	return loss

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

def pack(ordered_dict, labels):
	features = ordered_dict['DNA']
	return tf.expand_dims(features, axis=-1),  tf.one_hot(labels, depth=3)
	#return tf.stack(list(ordered_dict.values()), axis=-1), tf.one_hot(labels, depth=3)

def features(features, labels):
	return features

def labels(features, labels):
	return labels


if __name__ == '__main__':
	usage = '%s [-opt1, [-opt2, ...]] directory' % __file__
	parser = argparse.ArgumentParser(description='', formatter_class=RawTextHelpFormatter, usage=usage)
	parser.add_argument('directory', type=is_valid_file, help='input directory')
	parser.add_argument('-o', '--outfile', action="store", default=sys.stdout, type=argparse.FileType('w'), help='where to write output [stdout]')
	parser.add_argument('-k', '--kfold', action="store", default=0, type=int, help='which kfold')
	parser.add_argument('-t', '--trim', action="store", default=0, type=int, help='how many bases to trim off window ends')
	parser.add_argument('-r', '--reg', action="store_true", help='use kernel regularizer')
	args = parser.parse_args()

	#class_weight = {0:0.5, 1:2, 2:1, 3:1}

	filepath = args.directory + "_" + 'trim='+str(args.trim) + ',reg='+str(args.reg) + ',fold='+str(args.kfold) + '.ckpt'
	cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=filepath,save_weights_only=True,verbose=1)

	#args.activation = tf.keras.layers.LeakyReLU(alpha=0.01)
	#if os.path.isfile(filepath):
	#	model.load_weights(filepath).expect_partial()
	model = mm.create_model_conv2(args)
	print(model.summary())

	colnames =  ['TYPE', 'GC', 'DNA']
	selnames =  ['TYPE', 'DNA']
	
	tfiles = tf.data.experimental.make_csv_dataset(
		#compression_type    = 'GZIP',
		#file_pattern        = "viruses/train/win/GCF_000836805*",
		file_pattern        = args.directory + "/GCF_??????[^" +str(args.kfold) + "]??*",
		field_delim         = '\t',
		header              = False,
		column_names        = colnames,
		select_columns      = selnames,
		column_defaults     = [tf.int32, tf.string],
		shuffle             = True,
		num_parallel_reads  = 200,
		shuffle_buffer_size = 4000,
		batch_size          = 4000,
		num_epochs          = 1,
		sloppy				= True,
		label_name          = colnames[0]
		)
	tdata = tfiles.map(pack)

	'''
	import numpy as np
	np.set_printoptions(edgeitems=10)
	np.core.arrayprint._line_width = 180
	for feature in tdata.take(1):
		print( feature )
	exit()
	'''
	vfiles = tf.data.experimental.make_csv_dataset(
		#file_pattern        = "viruses/train/win/GCF_000836805*",
		file_pattern        = args.directory + "/GCF_??????[" +str(args.kfold) + "]??*",
		field_delim         = '\t',
		header              = False,
		column_names        = colnames,
		select_columns      = selnames,
		column_defaults     = [tf.int32, tf.string],
		batch_size          = 4000,
		num_epochs          = 1,
		shuffle             = True,
		num_parallel_reads  = 10,
		sloppy				= True,
		label_name          = colnames[0]
		#label_name          = 0 # this is old version
		)
	vdata = vfiles.map(pack)


	with tf.device('/device:GPU:0'):
		#es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, min_delta=0.001, baseline=None)
		#lr_callback = LRFinder()
		model.fit(
				  tdata,
				  validation_data = vdata,
				  epochs          = 20,
				  #class_weight    = class_weight,
				  verbose         = 0,
				  callbacks       = [LossHistoryCallback(), cp_callback]
		)
