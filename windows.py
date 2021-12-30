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

# Helper libraries
os.environ['OPENBLAS_NUM_THREADS'] = '9'
os.environ['MKL_NUM_THREADS'] = '9'
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

def pack(ordered_dict, labels):
	features = ordered_dict['DNA']
	return tf.expand_dims(features, axis=-1),  tf.one_hot(labels, depth=3)


if __name__ == '__main__':
	usage = '%s [-opt1, [-opt2, ...]] directory' % __file__
	parser = argparse.ArgumentParser(description='', formatter_class=RawTextHelpFormatter, usage=usage)
	parser.add_argument('directory', type=is_valid_file, help='input directory')
	parser.add_argument('-o', '--outfile', action="store", default=sys.stdout, type=argparse.FileType('w'), help='where to write output [stdout]')
	parser.add_argument('-k', '--kfold', action="store", default=0, type=int, help='which kfold')
	parser.add_argument('-t', '--trim', action="store", default=0, type=int, help='how many bases to trim off window ends')
	parser.add_argument('-r', '--reg', action="store_true", help='use kernel regularizer')
	args = parser.parse_args()

	class_weight = {0:1, 1:1, 2:1}

	filepath = args.directory + "_" + 'trim='+str(args.trim) + ',reg='+str(args.reg) + ',fold='+str(args.kfold) + ',weights=[%s,%s,%s]' % tuple(class_weight.values()) +'.ckpt'
	cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=filepath,save_weights_only=True,verbose=1)


	#if os.path.isfile(filepath):
	#	model.load_weights(filepath).expect_partial()
	model = mm.create_model_conv2(args)
	print(model.summary())

	colnames =  ['TYPE', 'GC', 'DNA']
	selnames =  ['TYPE', 'DNA']
	
	tfiles = tf.data.experimental.make_csv_dataset(
		#compression_type    = 'GZIP',
		file_pattern        = "viruses/train/win/GCF_000836805*",
		#file_pattern        = "viruses/train/win/GCF_??????[^" +str(args.kfold) + "]??*",
		field_delim         = '\t',
		header              = False,
		column_names        = colnames,
		select_columns      = selnames,
		column_defaults     = [tf.int32] + [tf.string],
		shuffle             = True,
		num_parallel_reads  = 200,
		shuffle_buffer_size = 5000,
		batch_size          = 10,
		num_epochs          = 1,
		sloppy				= True,
		label_name          = colnames[0]
		)

	tdata = tfiles.map(pack)

	'''
	for feature in tdata.take(1):
		print( feature )
	exit()
	'''

	vfiles = tf.data.experimental.make_csv_dataset(
		file_pattern        = "viruses/train/win/GCF_000836805*",
		#file_pattern        = "viruses/train/win/GCF_??????[" +str(args.kfold) + "]??*",
		field_delim         = '\t',
		header              = False,
		column_names        = colnames,
		select_columns      = selnames,
		column_defaults     = [tf.int32] + [tf.string],
		batch_size          = 10,
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
				  tdata.shard(num_shards=5, index=0),
				  validation_data = vdata,
				  epochs          = 3,
				  class_weight   = class_weight,
				  verbose         = 1,
				  callbacks       = [LossHistoryCallback(), cp_callback]
		)





