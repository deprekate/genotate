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
#from tensorflow.keras import mixed_precision
import keras_tuner as kt
#tf.keras.backend.set_floatx('float16')
import numpy as np

#mixed_precision.set_global_policy('mixed_float16')

'''
try:
  tpu = tf.distribute.cluster_resolver.TPUClusterResolver() # TPU detection
except ValueError:
  tpu = None
if tpu:
  policyConfig = 'mixed_bfloat16'
else:
  policyConfig = 'mixed_float16'
policy = tf.keras.mixed_precision.Policy(policyConfig)
tf.keras.mixed_precision.set_global_policy(policy)
'''
'''
gpus = tf.config.list_physical_devices('GPU')
if gpus:
	try:
		#tf.config.experimental.set_virtual_device_configuration(
        #	gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
  	  # Currently, memory growth needs to be the same across GPUs
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu, True)
		logical_gpus = tf.config.list_logical_devices('GPU')
		print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
	except RuntimeError as e:
		# Memory growth must be set before GPUs have been initialized
		print(e)
'''
#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#if len(physical_devices) > 0:
#	tf.config.experimental.set_memory_growth(physical_devices[0], True)

from genbank.file import File
from genotate.functions import parse_locus
from genotate.make_model import create_model_blend, blend, api, HyperRegressor



def is_valid_file(x):
	if not os.path.exists(x):
		raise argparse.ArgumentTypeError("{0} does not exist".format(x))
	return x

def pack(features): #labels,datas,windows): #features):
	#labels,windows,datas = tf.split(features, [1,99,3], axis=-1)
	labels,windows = tf.split(features, [1,99], axis=-1)
	#labels = tf.cast(labels, dtype=tf.int32)
	labels = tf.one_hot(labels, depth=3, axis=-1) #, dtype=tf.int32)
	#labels = tf.reshape(labels, [-1])
	labels = tf.squeeze(labels)
	#print(labels) ; print(datas) ; print(windows)
	#return (windows, datas) , labels
	return windows , labels

def rev_comp(seq):
	seq_dict = {'a':'t','t':'a','g':'c','c':'g',
                'n':'n',
                'r':'y','y':'r','s':'s','w':'w','k':'m','m':'k',
                'b':'v','v':'b','d':'h','h':'d'}
	return "".join([seq_dict[base] for base in reversed(seq)])


def iter_genbank(infile):
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


#np.set_printoptions(linewidth=500)
#np.set_printoptions(formatter={'all': lambda x: " {:.0f} ".format(x)})
#for row in iter_genbank('test/phiX174.fna'.encode()):
#	print(row)
#exit()

if __name__ == '__main__':
	usage = '%s [-opt1, [-opt2, ...]] directory' % __file__
	parser = argparse.ArgumentParser(description='', formatter_class=RawTextHelpFormatter, usage=usage)
	parser.add_argument('directory', type=is_valid_file, help='input directory')
	parser.add_argument('-o', '--outfile', action="store", default=sys.stdout, type=argparse.FileType('w'), help='where to write output [stdout]')
	parser.add_argument('-k', '--kfold', action="store", default=0, type=int, help='which kfold')
	parser.add_argument('-t', '--trim', action="store", default=0, type=int, help='how many bases to trim off window ends')
	parser.add_argument('-r', '--reg', action="store_true", help='use kernel regularizer')
	args = parser.parse_args()
	
	if args.kfold == 0:
		os.environ["CUDA_VISIBLE_DEVICES"]="0"
		os.environ["KERASTUNER_TUNER_ID"]="chief"
	else:
		#os.environ["CUDA_VISIBLE_DEVICES"]="0"
		os.environ["CUDA_VISIBLE_DEVICES"]=str(args.kfold%7+1)
		os.environ["KERASTUNER_TUNER_ID"]="tuner" + str(args.kfold)
	os.environ["KERASTUNER_ORACLE_IP"]="127.0.0.1"
	os.environ["KERASTUNER_ORACLE_PORT"]="8000"
	physical_devices = tf.config.experimental.list_physical_devices('GPU')
	tf.config.experimental.set_memory_growth(physical_devices[0], True)

	#filenames = [os.path.join(args.directory,f) for f in listdir(args.directory) if isfile(join(args.directory, f))]
	filenames = list()
	valnames = list()
	for f in listdir(args.directory):
		if f[9] not in '1357': 
			filenames.append(os.path.join(args.directory,f))
		else:
			valnames.append(os.path.join(args.directory,f))

	#filenames = filenames[:12] ; valnames = valnames[:12]
	print("Starting...")

	dataset = tf.data.Dataset.from_tensor_slices(filenames)
	dataset = dataset.interleave(
						lambda x: tf.data.Dataset.from_generator(
							iter_genbank,
							args=(x,),
							output_signature=(
								tf.TensorSpec(shape=(6,100),dtype=tf.int32)
							)
						),
						num_parallel_calls=tf.data.AUTOTUNE,
						#cycle_length=70,
						block_length=10
						)
	dataset = dataset.unbatch()
	dataset = dataset.map(pack)
	#dataset = dataset.shuffle(buffer_size=1000)
	dataset = dataset.batch(4096)
	dataset = dataset.prefetch(tf.data.AUTOTUNE)


	#for feature in dataset.take(1):
	#	print( feature )
	#exit()
	valset = tf.data.Dataset.from_tensor_slices(valnames)
	valset = valset.interleave(
						lambda x: tf.data.Dataset.from_generator(
							iter_genbank,
							args=(x,),
							output_signature=(
								tf.TensorSpec(shape=(6,100),dtype=tf.int32)
							)
						),
						num_parallel_calls=tf.data.AUTOTUNE,
						#cycle_length=70,
						block_length=10
						)
	valset = valset.unbatch().map(pack).batch(4096).prefetch(tf.data.AUTOTUNE)

	with tf.device('/device:GPU:0'):
		model = api(args)
		tuner = kt.Hyperband(HyperRegressor(),
					 hyperband_iterations=1000,
                     objective='val_accuracy',
					 #objective='val_loss',
                     max_epochs=10,
                     factor=3,
                     directory='final',
                     project_name='intro_to_kt')
		stop_early = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
		
		tuner.search(dataset, validation_data=valset, verbose=0) #, callbacks=[]) #stop_early])
		if args.kfold == 0:
			print(tuner.get_best_hyperparameters(1)[0].values)
			print(tuner.get_best_hyperparameters(2)[1].values)
			print(tuner.get_best_hyperparameters(3)[2].values)

