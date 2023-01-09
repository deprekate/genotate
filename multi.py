import os
import sys
import re
import argparse
from argparse import RawTextHelpFormatter
from os import listdir
from os.path import isfile, join
import datetime

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
import tensorflow as tf
#tf.keras.backend.set_floatx('float16')
import numpy as np


from genbank.file import File
from genotate.functions import *
from genotate.make_model import create_model_blend, blend, api

#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#if len(physical_devices) > 0:
#	tf.config.experimental.set_memory_growth(physical_devices[0], True)


def is_valid_file(x):
	if not os.path.exists(x):
		raise argparse.ArgumentTypeError("{0} does not exist".format(x))
	return x

def pack(features): #labels,datas,windows): #features):
	labels,datas,windows = tf.split(features, [1,3,99], axis=-1)
	labels = tf.cast(labels, dtype=tf.int32)
	labels = tf.one_hot(labels, depth=3, axis=-1) #, dtype=tf.int32)
	#labels = tf.reshape(labels, [-1])
	labels = tf.squeeze(labels)
	#print(labels) ; print(datas) ; print(windows)
	return (windows, datas) , labels

def rev_comp(seq):
	seq_dict = {'a':'t','t':'a','g':'c','c':'g',
                'n':'n',
                'r':'y','y':'r','s':'s','w':'w','k':'m','m':'k',
                'b':'v','v':'b','d':'h','h':'d'}
	return "".join([seq_dict[base] for base in reversed(seq)])


def parse_genbank(infile):
	genbank = File(infile.decode())
	for locus in genbank:
		for data in parse_locus(locus):
			yield data

class LossHistoryCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, batch, logs=None):
		#logs['loss'] or logs['val_loss'] (the latter available only if you use validation data when model.fit()
		# Use logs['loss'] or logs['val_loss'] for pyqt5 purposes
		row = list()
		for key, value in logs.items():
			row.append(key)
			row.append(value)
		print('\t'.join(map(str,row)), flush=True)


if __name__ == '__main__':
	usage = '%s [-opt1, [-opt2, ...]] directory' % __file__
	parser = argparse.ArgumentParser(description='', formatter_class=RawTextHelpFormatter, usage=usage)
	parser.add_argument('directory', type=is_valid_file, help='input directory')
	parser.add_argument('-o', '--outfile', action="store", default=sys.stdout, type=argparse.FileType('w'), help='where to write output [stdout]')
	parser.add_argument('-k', '--kfold', action="store", default=0, type=int, help='which kfold')
	parser.add_argument('-t', '--trim', action="store", default=0, type=int, help='how many bases to trim off window ends')
	parser.add_argument('-r', '--reg', action="store_true", help='use kernel regularizer')
	args = parser.parse_args()

	filenames = [os.path.join(args.directory,f) for f in listdir(args.directory) if isfile(join(args.directory, f))]
	print("Starting...")
	with tf.device('/device:GPU:0'):
		dataset = tf.data.Dataset.from_tensor_slices(filenames)
		dataset = dataset.interleave(
							lambda x: tf.data.Dataset.from_generator(
								parse_genbank,
								args=(x,),
								output_signature=(
									tf.TensorSpec(shape=(6,103),dtype=tf.float32)
								)
							),
							num_parallel_calls=tf.data.AUTOTUNE,
							#cycle_length=70,
							block_length=10
							)
		dataset = dataset.unbatch()
		dataset = dataset.map(pack)
		#dataset = dataset.shuffle(buffer_size=1000)
		dataset = dataset.batch(4000)
		dataset = dataset.prefetch(tf.data.AUTOTUNE)

		log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
		tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch = '1000,1010')
	
		cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="multi",save_weights_only=True,verbose=1)

		#for feature in dataset.take(1):
		#	print( feature )
		#exit()

		model = blend(args)
		#model = create_model_blend(args)
		model.fit(
			dataset,
			epochs          = 3,
			verbose         = 1,
			callbacks=[ cp_callback] #, LossHistoryCallback() ] #,tensorboard_callback]
		)
