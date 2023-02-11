import os
import sys
import re
import argparse
from argparse import RawTextHelpFormatter
from os import listdir
from os.path import isfile, join
import datetime
import time

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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

def iter_genbank(infile):
	for locus in File(infile.decode()):
		# label the positions
		positions = dict()
		for feature in locus.features(include=['CDS']):
			for i,*_ in feature.codon_locations():
				# do the other 5 frames
				for sign,offset in [(+1,1), (+1,2), (-1,1), (-1,2), (-1,0)]:
					pos = sign * (i + offset) * feature.strand
					if pos not in positions:
						positions[pos] = 0
				# do the current frame
				sign,offset = (+1,0)
				pos = sign * (i + offset) * feature.strand
				positions[pos] = 1
		dna = locus.seq()
		forward = np.zeros(48+len(dna)+50)
		reverse = np.zeros(48+len(dna)+50)
		for i,base in enumerate(dna):
			#if base in 'acgt':
			forward[i+48] = ((ord(base) >> 1) & 3) + 1
			reverse[i+48] = ((forward[i+48] - 3) % 4) + 1
		a = np.zeros([6, 100], dtype=int)
		# leave this here for numpy < 1.20 backwards compat
		#forfor = np.concatenate(( forward, forward[:-1] ))
		#L = len(forward)
		#n = forfor.strides[0]
		#f = np.lib.stride_tricks.as_strided(forfor[L-1:], (L,L), (-n,n))
		w = np.lib.stride_tricks.sliding_window_view(a,99)
		yield w

def is_valid_file(x):
	if not os.path.exists(x):
		raise argparse.ArgumentTypeError("{0} does not exist".format(x))
	return x

def pack(labels, windows): #labels,datas,windows): #features):
	#labels,windows,datas = tf.split(features, [1,99,3], axis=-1)
	#labels,windows = tf.split(features, [1,99], axis=-1)
	#windows = tf.unstack(windows)
	#labels = tf.cast(labels, dtype=tf.int32)
	labels = tf.one_hot(labels, depth=3, axis=-1) #, dtype=tf.int32)
	#labels = tf.squeeze(labels)
	#labels = tf.reshape(labels, [-1])
	#labels = tf.unstack(labels)
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
#for row in iter_genbank(sys.argv[1].encode()):
#	for i in range(6):
#		print(row[i,0], to_dna(row[i,1:].tolist()))
#		exit()

from genotate.fun import GenomeDataset

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
	for f in listdir(args.directory):
		if (int(f[11])%5) != args.kfold - 1: 
			filenames.append(os.path.join(args.directory,f))
		else:
			valnames.append(os.path.join(args.directory,f))
	'''
	print(filenames)
	print(len(filenames))
	print(valnames)
	print(len(valnames))
	print()
	'''
	#filenames = filenames[:10] ; valnames = valnames[:10]
	print("Starting...",flush=True)
	with tf.device('/device:GPU:0'):
		model = api(args)
		#model = blend(args)
	dataset = tf.data.Dataset.from_tensor_slices(filenames)
	dataset = dataset.interleave(
						#lambda x: GenomeDataset(x).unbatch(),
						lambda x: GenomeDataset(x), #.flat_map(tf.data.Dataset.from_tensor_slices),
						#lambda x: tf.data.Dataset.from_generator(
						#	iter_genbank,
						#	args=(x,),
						#	output_signature=(
						#		tf.TensorSpec(shape=(6,100),dtype=tf.int32)
						#	)
						#),
						num_parallel_calls=tf.data.AUTOTUNE,
						cycle_length=64,
						block_length=6,
						deterministic=False
						)
	#dataset = dataset.unbatch()
	#dataset = dataset.cache()
	#dataset = dataset.shuffle(buffer_size=1000)
	dataset = dataset.map(pack, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
	dataset = dataset.batch(4096, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
	dataset = dataset.prefetch(tf.data.AUTOTUNE)
	
	#for feature in dataset.take(1):
	#	print( feature[0].shape )
	#	print( feature )
	#exit()

	valset = tf.data.Dataset.from_tensor_slices(valnames)
	valset = valset.interleave(
						lambda x: GenomeDataset(x), #.flat_map(tf.data.Dataset.from_tensor_slices),
						#lambda x: tf.data.Dataset.from_generator(
						#	iter_genbank,
						#	args=(x,),
						#	output_signature=(
						#		tf.TensorSpec(shape=(6,100),dtype=tf.int32)
						#	)
						#),
						num_parallel_calls=tf.data.AUTOTUNE,
						cycle_length=64,
						block_length=6,
						deterministic=False
						)
	#valset = valset.unbatch().map(pack).batch(4096).prefetch(tf.data.AUTOTUNE)
	valset = valset.map(pack).batch(4096).prefetch(tf.data.AUTOTUNE)
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), histogram_freq=1, profile_batch = '1512,2024')

	#cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="phage_fold-"+args.kfold ,save_weights_only=True,verbose=1)
	checkpoint = tf.keras.callbacks.ModelCheckpoint('phage_' + str(args.kfold) + 'fold-{epoch:03d}', save_weights_only=True, save_freq='epoch')
	es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)

	print(dataset)
	#model = create_model_blend(args)
	model.fit(
		dataset,
		#validation_data = valset,
		epochs          = 100,
		verbose         = 1,
		callbacks=[ checkpoint, es_callback, LossHistoryCallback() ,tensorboard_callback]
	)
