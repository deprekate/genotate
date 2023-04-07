import os
import sys
import argparse
from argparse import RawTextHelpFormatter
from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE,SIG_DFL)
import socket

os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ["OMP_NUM_THREADS"]="4" 
#os.environ['TF_GPU_THREAD_COUNT'] = '4'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import keras_tuner as kt
from genotate.make_model import create_model_blend, blend, api, HyperRegressor
import numpy as np
from genbank.file import File
from genotate.functions import to_dna, getsize
import datetime
import time

def is_valid_file(x):
	if not os.path.exists(x):
		raise argparse.ArgumentTypeError("{0} does not exist".format(x))
	return x

class LossHistoryCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, batch, logs=None):
		#logs['loss'] or logs['val_loss'] (the latter available only if you use validation data when model.fit()
		# Use logs['loss'] or logs['val_loss'] for pyqt5 purposes
		row = list()
		for key, value in logs.items():
			row.append(key)
			row.append(value)
		print('\t'.join(map(str,row)), flush=True)

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
				#s = (feature.strand * -1 + 1) >> 1
				s = (feature.strand >> 1) * -1
				locations = feature.codon_locations()
				if feature.partial() == 'left': next(locations)
				for i,*_ in locations:
					i = 2 * i
					# coding
					Y[i+s  ,1] = 1
					# noncoding
					Y[i:i+6,0] = 1
					Y[i:i+6,2] = 0
				locus[feature] = None
			Y[Y[:,1]==1,0] = 0
			forward = np.zeros((w//2-1)+length+(w//2+1),dtype=np.uint8)
			reverse = np.zeros((w//2-1)+length+(w//2+1),dtype=np.uint8)
			for i,base in enumerate(locus.dna):
				i += w//2-1
				#if base in 'acgt':
				forward[i] = ((ord(base) >> 1) & 3) + 1
				reverse[i] = ((forward[i] - 3) % 4) + 1
			locus.dna = None
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
			# leave this here for numpy < 1.20 backwards compat
			#forfor = np.concatenate(( forward, forward[:-1] ))
			#L = len(forward)
			#n = forfor.strides[0]
			#f = np.lib.stride_tricks.as_strided(forfor[L-1:], (L,L), (-n,n))
			# this is creates one BIG numpy array
			'''
			X = np.zeros([len(locus.dna)*2  ,99],dtype=np.uint8)
			X[0::2,] = np.lib.stride_tricks.sliding_window_view(forward,99)
			X[1::2,] = np.lib.stride_tricks.sliding_window_view(reverse,99)[:,::-1]
			#X[:,0:9] = [int(s) for s in self.name.decode()[-19:-10]]
			#A[I:i+1,:] = w
			#I = i
			yield X,Y[:-7]
			'''
			# this splits the BIG numpy array into n sized chunks to limit ram usage
			for i in range( length // n):
				i *= n
				X[0::2,] = np.lib.stride_tricks.sliding_window_view(forward[ i : i+n+w-1],w)
				X[1::2,] = np.lib.stride_tricks.sliding_window_view(reverse[ i : i+n+w-1],w)[:,::-1]
				yield X,Y[ i*2:i*2+n*2 , :]
			i = length // n * n
			r = length % n
			if r:
				X[0:2*r:2,] = np.lib.stride_tricks.sliding_window_view(forward[ i : i+r+w-1],w)
				X[1:2*r:2,] = np.lib.stride_tricks.sliding_window_view(reverse[ i : i+r+w-1],w)[:,::-1]
				yield X[ : 2*r , : ] , Y[ i*2: i*2+2*r , :]

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
	args = parser.parse_args()

	if args.kfold == -1:
		os.environ["CUDA_VISIBLE_DEVICES"]="0"
		os.environ["KERASTUNER_TUNER_ID"]="chief"
	else:
		#os.environ["CUDA_VISIBLE_DEVICES"]="0"
		os.environ["CUDA_VISIBLE_DEVICES"]=str(args.kfold)
		os.environ["KERASTUNER_TUNER_ID"]="tuner" + str(abs(hash(socket.gethostname()))) + str(args.kfold)
	os.environ["KERASTUNER_ORACLE_IP"]="127.0.0.1"
	os.environ["KERASTUNER_ORACLE_PORT"]="8000"
	'''
	os.environ["CUDA_VISIBLE_DEVICES"]="0"
	'''
	gpus = tf.config.list_physical_devices('GPU')
	for gpu in gpus:
		tf.config.experimental.set_memory_growth(gpu, True)

	#filenames = [os.path.join(args.directory,f) for f in os.listdir(args.directory) if os.path.isfile(os.path.join(args.directory, f))]
	filenames = list()
	valnames = list()
	for f in os.listdir(args.directory):
		#if (int(f[11])%5) != args.kfold: 
		if f[11] not in '357': 
			filenames.append(os.path.join(args.directory,f))
		else:
			valnames.append(os.path.join(args.directory,f))
	#filenames = filenames[:10] ; valnames = valnames[:10]
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
					cycle_length=192,
					block_length=48,
	                )
	#dataset = dataset.unbatch()
	dataset = dataset.shuffle(buffer_size=64)
	dataset = dataset.batch(9216*len(gpus), num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
	dataset = dataset.shuffle(buffer_size=8)
	dataset = dataset.prefetch(tf.data.AUTOTUNE)
	#dataset = strategy.experimental_distribute_dataset(dataset)

	options = tf.data.Options()
	options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.FILE
	dataset = dataset.with_options(options) 

	#print(dataset)
	#for x,y in dataset.take(1):
	#	x = x.numpy()
	#	y = y.numpy()
	#	for i in range(len(x)):
	#		print(y[i,:],x[i,:])
	#		print(to_dna(x[i,10:]))
	#exit()
	
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
					cycle_length=18,block_length=512,
	                )
	valiset = valiset.batch(9216*len(gpus), num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
	valiset = valiset.prefetch(tf.data.AUTOTUNE)
	#valiset = strategy.experimental_distribute_dataset(valiset)
	valiset = valiset.with_options(options) 

	'''
	name = '_'.join(os.path.dirname(args.directory).split('/')[-2:])
	checkpoint = tf.keras.callbacks.ModelCheckpoint(name + str(args.kfold) + '-{epoch:03d}', save_weights_only=True, save_freq='epoch')
	es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=4)
	save = tf.keras.callbacks.BackupAndRestore(name+str(args.kfold)+"_backup", save_freq="epoch", delete_checkpoint=True)
	#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), histogram_freq=1, profile_batch = '1512,2024')
	with strategy.scope():
	#with tf.device('/device:GPU:0'):
		model = api(None)
	model.fit(
		dataset,
		validation_data = valiset,
		epochs          = 10,
		verbose         = 1,
		callbacks       = [LossHistoryCallback(), checkpoint, es_callback, save] #, checkpoint, LossHistoryCallback() ] #tensorboard_callback]
	)
	'''
	with tf.device('/device:GPU:0'):
		model = api(args)
		tuner = kt.Hyperband(HyperRegressor(),
					 hyperband_iterations=3,
                     objective='val_accuracy',
					 #objective='val_loss',
                     max_epochs=15,
                     factor=3,
                     directory='tuner',
                     project_name='phages')
		stop_early = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
		
		tuner.search(dataset, validation_data=valiset, verbose=1) #, callbacks=[]) #stop_early])
		if args.kfold == 0:
			print(tuner.get_best_hyperparameters(1)[0].values)
			print(tuner.get_best_hyperparameters(2)[1].values)
			print(tuner.get_best_hyperparameters(3)[2].values)
