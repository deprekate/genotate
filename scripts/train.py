import os
import sys
import argparse
from argparse import RawTextHelpFormatter
from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE,SIG_DFL)
import threading
from packaging import version
import socket
import gc

os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ["OMP_NUM_THREADS"]="8" 
#os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] ='true'
#os.environ['TF_GPU_THREAD_COUNT'] = '4'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
#import keras_tuner as kt
from genotate.make_model import create_model_blend, blend, api #, HyperRegressor
import numpy as np
from genbank.file import File
from genotate.functions import to_dna, getsize
import datetime
import time
from statistics import stdev

# backward compatibility
def sliding_window_view(ar, i):
	sys.stdout.write("using legacy sliding window")
	a = np.concatenate(( ar, ar[:-1] ))
	L = len(ar)
	n = a.strides[0]
	return np.lib.stride_tricks.as_strided(a, (L,L), (n,n), writeable=False)[:-i+1,:i]
if version.parse(np.__version__) < version.parse('1.20'):
	setattr(np.lib.stride_tricks, 'sliding_window_view', sliding_window_view)


class optChanger(tf.keras.callbacks.Callback):
	def __init__(self):
		self.loss = [1,1,1]
		self.opt = 'Adam'
	def on_epoch_end(self, epoch, logs=None):
		self.loss.append(logs['loss'])
		print('opt',self.model.optimizer)
		if self.loss[-1] >  self.loss[-2]:
			self.model.optimizer = tf.keras.optimizers.SGD()
			self.loss.extend([1,1,1])
			self.opt = 'SGD'
			return
		if self.opt == 'SGD' and round(min(self.loss[-3:]),4) == round(self.loss[-3],4):
				print('early stop')
				self.stopped_epoch = epoch
				self.model.stop_training = True

def is_valid_file(x):
	if not os.path.exists(x):
		raise argparse.ArgumentTypeError("{0} does not exist".format(x))
	return x

class LossHistoryCallback(tf.keras.callbacks.Callback):
	def __init__(self, name, data):
		self.name = name
		self.data = data
	def on_epoch_end(self, epoch, logs=None):
		#print(self.name, epoch, self.data)
		#logs['loss'] or logs['val_loss'] (the latter available only if you use validation data when model.fit()
		# Use logs['loss'] or logs['val_loss'] for pyqt5 purposes
		row = [epoch+1]
		for key, value in logs.items():
			row.append(key)
			row.append(value)
		print('\t'.join(map(str,row)), flush=True)
		#gc.collect()
		#tf.keras.backend.clear_session()	
		#x = threading.Thread(target=validate, args=(self.name+'-'+str(epoch+1).rjust(3,'0') ,self.data))
		#x.start()

class GenomeDataset:
	#@tf.function
	def __init__(self, filename):
		#spec = (tf.TensorSpec(shape = (None,99), dtype = tf.experimental.numpy.int8),
		#		tf.TensorSpec(shape = (None, 3), dtype = tf.experimental.numpy.int8))
		self.name = filename
		self.file = File(filename.decode())
	def __iter__(self):
		w = 87
		n = 20000
		X = np.zeros([2*n ,w],dtype=np.uint8)
		for locus in self.file:
			length = len(locus.dna)
			Y = np.zeros([length*2+7,3],dtype=np.uint8)
			Y[:,2] = 1
			# label the positions
			for feature in locus.features(include=['CDS']):
				s = (feature.strand >> 1) * -1
				#if feature.partial() == 'left': next(locations)
				for loc in feature.codon_locations():
					i = 2 * loc[2*s]
					# coding
					Y[i+s  ,1] = 1
					# noncoding
					Y[i:i+6,0] = 1
					Y[i:i+6,2] = 0
				locus[feature] = None
			#Y[Y[:,1]==1,0] = 0
			Y[:, 0][Y[:, 1] == 1] = 0
			forward = np.zeros((w//2-1)+length+(w//2+1),dtype=np.uint8)
			reverse = np.zeros((w//2-1)+length+(w//2+1),dtype=np.uint8)
			#for i,base in enumerate(locus.dna):
			#	i += w//2-1
				#if base in 'acgt':
			#	forward[i] = ((ord(base) >> 1) & 3) + 1
			#	reverse[i] = ((forward[i] - 3) % 4) + 1
			forward[w//2-1 : -w//2] = (np.frombuffer(locus.dna.encode(), dtype=np.uint8) >> 1 & 3) + 1
			reverse[w//2-1 : -w//2] = ((forward[w//2-1:-w//2] - 3) % 4) + 1
			locus.dna = None
			for i in range( length // n):
				i *= n
				X[0::2,] = np.lib.stride_tricks.sliding_window_view(forward[ i : i+n+w-1],w)
				X[1::2,] = np.lib.stride_tricks.sliding_window_view(reverse[ i : i+n+w-1],w)[:,::-1]
				yield X,Y[ i*2:i*2+n*2 , :]
			i = length // n * n
			r = length % n
			if r:
				X[0:2*r:2,] = np.lib.stride_tricks.sliding_window_view(forward[i : i+r+w-1],w)
				X[1:2*r:2,] = np.lib.stride_tricks.sliding_window_view(reverse[i : i+r+w-1],w)[:,::-1]
				yield X[ : 2*r , : ] , Y[ i*2: i*2+2*r , :]
			'''
			a = np.zeros([6, 99], dtype=np.uint8)
			for n in range(0, length-2, 3):
				#pos = n//100
				i = n + 0
				a[0,0:99] = forward[i : i+99 ]
				a[1,0:99] = reverse[i : i+99 ][::-1]
				i = n + 1
				a[2,0:99] = forward[i : i+99 ]
				a[3,0:99] = reverse[i : i+99 ][::-1]
				i = n + 2
				a[4,0:99] = forward[i : i+99 ]
				a[5,0:99] = reverse[i : i+99 ][::-1]
				yield a,Y[2*n:2*n+6,:]
			'''
			'''
			X = np.zeros([len(locus.dna)*2  ,99],dtype=np.uint8)
			X[0::2,] = np.lib.stride_tricks.sliding_window_view(forward,99)
			X[1::2,] = np.lib.stride_tricks.sliding_window_view(reverse,99)[:,::-1]
			#X[:,0:9] = [int(s) for s in self.name.decode()[-19:-10]]
			#A[I:i+1,:] = w
			#I = i
			yield X,Y[:-7]
			'''

'''
dataset = GenomeDataset(sys.argv[1].encode())
#print(getsize(dataset))
#dataset = GenomeDataset("/home/katelyn/develop/genotate/test/NC_001416.gbk".encode())
n = 0
for x,y in dataset:
	#print(x.shape, y.shape)
#	break
	for i in range(len(x)):
		print(n//2, y[i,:], to_dna(x[i,:]))		
		n += 1
exit()
#print(getsize(dataset))
#exit()
'''


if __name__ == '__main__':
	usage = '%s [-opt1, [-opt2, ...]] directory' % __file__
	parser = argparse.ArgumentParser(description='', formatter_class=RawTextHelpFormatter, usage=usage)
	parser.add_argument('directory', type=is_valid_file, help='input directory')
	parser.add_argument('-o', '--outfile', action="store", default=sys.stdout, type=argparse.FileType('w'), help='where to write output [stdout]')
	parser.add_argument('-k', '--kfold', action="store", default=0, type=int, help='which kfold')
	parser.add_argument('-t', '--trim', action="store", default=0, type=int, help='how many bases to trim off window ends')
	parser.add_argument('-r', '--reg', action="store_true", help='use kernel regularizer')
	parser.add_argument('-n', '--none', action='store', type=str)
	args = parser.parse_args()

	os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
	gpus = tf.config.list_physical_devices('GPU')
	for gpu in gpus:
		tf.config.experimental.set_memory_growth(gpu, True)
		#tf.config.set_logical_device_configuration(gpu,[tf.config.LogicalDeviceConfiguration(memory_limit=8192)])

	#filenames = [os.path.join(args.directory,f) for f in os.listdir(args.directory) if os.path.isfile(os.path.join(args.directory, f))]
	filenames = list()
	valnames = list()
	for f in os.listdir(args.directory):
		if (int(f[11])%5) != args.kfold: 
		#if f[11] not in '357': 
		#if (int(f[11])%2) != args.kfold: 
			filenames.append(os.path.join(args.directory,f))
		else:
			valnames.append(os.path.join(args.directory,f))
	#filenames = filenames[:7] ; valnames = valnames[:7]
	#print(filenames) ; print(valnames)
	print(len(filenames)) ; print(len(valnames))
	spec = (tf.TensorSpec(shape = (None,87), dtype = tf.experimental.numpy.int8),
			tf.TensorSpec(shape = (None, 3), dtype = tf.experimental.numpy.int8))
	strategy = tf.distribute.MirroredStrategy(devices=[item.name.replace('physical_device:','').lower() for item in gpus])
	dataset = tf.data.Dataset.from_tensor_slices(filenames)
	dataset = dataset.shuffle(buffer_size=64)
	dataset = dataset.interleave(
					#lambda x: GenomeDataset(x).unbatch(),
					#lambda x: tf.data.Dataset.from_tensor_slices(GenomeDataset(x)).unbatch(),
					lambda x: tf.data.Dataset.from_generator(
						GenomeDataset,
						args=(x,),
						output_signature=spec
					).unbatch(),
				    num_parallel_calls=tf.data.AUTOTUNE,
					deterministic=False,
					cycle_length=180,
					block_length=48,
				    )
	#dataset = dataset.unbatch()
	dataset = dataset.shuffle(buffer_size=1024)
	dataset = dataset.batch(8640*len(gpus), num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
	dataset = dataset.shuffle(buffer_size=8)
	dataset = dataset.prefetch(tf.data.AUTOTUNE)
	#dataset = strategy.experimental_distribute_dataset(dataset)
	
	options = tf.data.Options()
	options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.FILE
	dataset = dataset.with_options(options) 

	valiset = tf.data.Dataset.from_tensor_slices(valnames)
	valiset = valiset.interleave(
					#lambda x: GenomeDataset(x).unbatch(),
					lambda x: tf.data.Dataset.from_generator(
						GenomeDataset,
						args=(x,),
						output_signature=spec
					).unbatch(),
				    num_parallel_calls=tf.data.AUTOTUNE,
					deterministic=False,
					cycle_length=16,block_length=540,
				    )
	valiset = valiset.batch(8640*len(gpus), num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
	valiset = valiset.prefetch(tf.data.AUTOTUNE)
	#valiset = strategy.experimental_distribute_dataset(valiset)
	valiset = valiset.with_options(options) 

	#print(dataset)
	#for x,y in dataset.take(1):
	#	x = x.numpy()
	#	y = y.numpy()
	#	for i in range(len(x)):
	#		print(y[i,:],x[i,:])
	#		print(to_dna(x[i,10:]))
	#exit()
	
	name = '_'.join(os.path.dirname(args.directory).split('/')[-2:]) + str(args.kfold)
	checkpoint = tf.keras.callbacks.ModelCheckpoint('dual/' + name + '-{epoch:03d}', save_weights_only=True, save_freq='epoch')
	#es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)
	#save = tf.keras.callbacks.BackupAndRestore(name+str(args.kfold)+"_backup", save_freq="epoch", delete_checkpoint=True)
	#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), histogram_freq=1, profile_batch = '1512,2024')
	with strategy.scope():
	#with tf.device('/device:GPU:0'):
		model = api(None)
	#model.load_weights('models/' + name + '-' + str(99).rjust(3,'0')).expect_partial()
	#model.load_weights('dual/assembly_phages0-050').expect_partial()
	model.fit(
		dataset,
		validation_data = valiset,
		epochs          = 300,
		verbose         = 2,
		callbacks       = [optChanger(), checkpoint] #, LossHistoryCallback(name, None)] #valiset) ] #es_callback] #tensorboard_callback]
	)
