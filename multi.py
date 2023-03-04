import os
import sys
import re
import argparse
from argparse import RawTextHelpFormatter
from os.path import isfile, join
import datetime
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
import tensorflow as tf
#from tensorflow.keras import mixed_precision
#tf.keras.backend.set_floatx('float16')
import numpy as np
time.sleep(1)
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

from genbank.file import File
from genotate.functions import parse_locus, to_dna 
from genotate.make_model import create_model_blend, blend, api

#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#if len(physical_devices) > 0:
#	tf.config.experimental.set_memory_growth(physical_devices[0], True)


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
		locus.file = infile.decode()	
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

np.set_printoptions(linewidth=500)

dataset = iter_genbank("/data/katelyn/assembly/phages/train/GCA_000840865.2.gbff.gz".encode())
n = 0
for xy in dataset:
	#print(x.shape, y.shape)
#	break
	for i in range(len(xy)):
		tem = np.zeros(3, dtype=np.uint8)
		tem[xy[i,0]] = 1
		print(n//2, tem, to_dna(xy[i,1:]))
		n += 1
exit()


#np.set_printoptions(formatter={'all': lambda x: " {:.0f} ".format(x)})
#for row in iter_genbank(sys.argv[1].encode()):
#	for i in range(6):
#		print(row[i,0], to_dna(row[i,1:].tolist()))
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

	#os.environ["CUDA_VISIBLE_DEVICES"]=str(args.kfold+2)
	physical_devices = tf.config.experimental.list_physical_devices('GPU')
	tf.config.experimental.set_memory_growth(physical_devices[0], True)


	#filenames = [os.path.join(args.directory,f) for f in listdir(args.directory) if isfile(join(args.directory, f))]
	filenames = list()
	valnames = list()
	for f in sorted(os.listdir(args.directory)):
		if (int(f[11])%5) != args.kfold: 
			filenames.append(os.path.join(args.directory,f))
		else:
			valnames.append(os.path.join(args.directory,f))
	#print(filenames) ; print(valnames)
	print(len(filenames)) ; print(len(valnames))
	
	#filenames = filenames[:10] ; valnames = valnames[:10]
	print("Starting...",flush=True)
	with tf.device('/device:GPU:0'):
		model = api(args)
		#model = blend(args)
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
						deterministic=True,
						cycle_length=64,
						block_length=8
						)
	dataset = dataset.unbatch()
	dataset = dataset.map(pack)
	dataset = dataset.batch(4096, deterministic=True)
	#dataset = dataset.shuffle(buffer_size=1000)
	print(dataset)
	for x,y in dataset.take(1):
		x = x.numpy()
		y = y.numpy()
		for i in range(len(x)):
			print(y[i,:], x[i,:])
		
	exit()
	dataset = dataset.prefetch(tf.data.AUTOTUNE)

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
						cycle_length=64,
						block_length=8
						)
	valset = valset.unbatch().map(pack).batch(4096).prefetch(tf.data.AUTOTUNE)
	#log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch = '1512,2024')

	#cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="phage_fold-"+args.kfold ,save_weights_only=True,verbose=1)
	checkpoint = tf.keras.callbacks.ModelCheckpoint('phage_' + str(args.kfold) + 'fold-{epoch:03d}', save_weights_only=True, save_freq='epoch')
	es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)


	#for feature in dataset.take(1):
	#	print( feature )
	#exit()

	#model = create_model_blend(args)
	model.fit(
		dataset,
		validation_data = valset,
		epochs          = 100,
		verbose         = 0,
		callbacks=[ checkpoint, es_callback, LossHistoryCallback() ] # ,tensorboard_callback]
	)
