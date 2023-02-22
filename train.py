import os
import sys
import argparse
from argparse import RawTextHelpFormatter
from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE,SIG_DFL)

#os.environ["OMP_NUM_THREADS"]="16" 
#os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_GPU_THREAD_COUNT'] = '4'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from genotate.make_model import create_model_blend, blend, api
import tensorflow as tf
import numpy as np
from genbank.file import File
from genotate.functions import parse_locus, to_dna
import datetime
import time


#os.environ["CUDA_VISIBLE_DEVICES"]="3" #,4,5,7"
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

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
		self.file = File(filename.decode())
	def __iter__(self):
		n = 10000
		X = np.zeros([2*n ,99],dtype=np.uint8)
		for locus in self.file:
			dna = locus.seq()
			Y = np.zeros([len(dna)*2+7,3],dtype=np.uint8)
			Y[:,2] = 1
			# label the positions
			positions = dict()
			for feature in locus.features(include=['CDS']):
				s = (feature.strand * -1 + 1) >> 1
				locations = feature.codon_locations()
				if feature.partial() == 'left': next(locations)
				for i in locations:
					# coding
					Y[2*i[0]+0+s,1] = 1
					# noncoding
					Y[2*i[0]+0+s,0] = 1
					Y[2*i[0]+1+s,0] = 1
					Y[2*i[0]+2+s,0] = 1
					Y[2*i[0]+3+s,0] = 1
					Y[2*i[0]+4+s,0] = 1
					Y[2*i[0]+5+s,0] = 1
					Y[2*i[0]+0+s,2] = 0
					Y[2*i[0]+1+s,2] = 0
					Y[2*i[0]+2+s,2] = 0
					Y[2*i[0]+3+s,2] = 0
					Y[2*i[0]+4+s,2] = 0
					Y[2*i[0]+5+s,2] = 0
			Y[Y[:,1]==1,0] = 0
			forward = np.zeros(48+len(dna)+50,dtype=np.uint8)
			reverse = np.zeros(48+len(dna)+50,dtype=np.uint8)
			for i,base in enumerate(dna):
				#if base in 'acgt':
				forward[i+48] = ((ord(base) >> 1) & 3) + 1
				reverse[i+48] = ((forward[i+48] - 3) % 4) + 1
			# leave this here for numpy < 1.20 backwards compat
			#forfor = np.concatenate(( forward, forward[:-1] ))
			#L = len(forward)
			#n = forfor.strides[0]
			#f = np.lib.stride_tricks.as_strided(forfor[L-1:], (L,L), (-n,n))
			# this is creates one BIG numpy array
			'''
			X = np.zeros([len(dna)*2  ,99],dtype=np.uint8)
			X[0::2,] = np.lib.stride_tricks.sliding_window_view(forward,99)
			X[1::2,] = np.lib.stride_tricks.sliding_window_view(reverse,99)[:,::-1]
			#A[I:i+1,:] = w
			#I = i
			yield X,Y[:-7]
			'''
			# this splits the BIG numpy array into n sized chunks to limit ram usage
			for i in range( len(dna) // n):
				X[0::2,] = np.lib.stride_tricks.sliding_window_view(forward[ i*n : i*n+n+98],99)
				X[1::2,] = np.lib.stride_tricks.sliding_window_view(reverse[ i*n : i*n+n+98],99)[:,::-1]
				yield X,Y[ i*2*n:i*2*n+2*n , :]
			i = len(dna) // n * n
			r = len(dna) % n
			X[0:2*r:2,] = np.lib.stride_tricks.sliding_window_view(forward[ i : i+r+98],99)
			X[1:2*r:2,] = np.lib.stride_tricks.sliding_window_view(reverse[ i : i+r+98],99)[:,::-1]
			yield X[ : 2*r , : ],Y[ i*2: i*2+2*r , :]


#dataset = GenomeDataset("/data/katelyn/assembly/bacteria/train/GCA_000005825.2.gbff.gz".encode())
#dataset = GenomeDataset("/home/katelyn/develop/genotate/test/NC_001416.gbk".encode())
#for x,y in dataset:
	#print(x.shape, y.shape)
	#for i in range(len(x)):
	#	print(y[i,:], to_dna(x[i,:]))		
#exit()
#for window,label in dataset.take(1):
#	print(window)
#	print(label, to_dna(window.numpy()))
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

	options = tf.data.Options()
	options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.FILE
	#filenames = [os.path.join(args.directory,f) for f in os.listdir(args.directory) if os.path.isfile(os.path.join(args.directory, f))]
	filenames = list()
	valnames = list()
	for f in os.listdir(args.directory):
		if (int(f[11])%5) != args.kfold: 
			filenames.append(os.path.join(args.directory,f))
		else:
			valnames.append(os.path.join(args.directory,f))
	#filenames = filenames[:10] ; valnames = valnames[:5]
	print(len(filenames)) ; print(len(valnames))
	
	spec = (tf.TensorSpec(shape = (None,99), dtype = tf.experimental.numpy.int8),
			tf.TensorSpec(shape = (None, 3), dtype = tf.experimental.numpy.int8))
	dataset = tf.data.Dataset.from_tensor_slices(filenames)
	dataset = dataset.interleave(
					#lambda x: GenomeDataset(x).unbatch(),
					lambda x: tf.data.Dataset.from_generator(
						GenomeDataset,
						args=(x,),
						output_signature=spec
					).unbatch(),
					#).rebatch(9216),
	                num_parallel_calls=tf.data.AUTOTUNE,
					deterministic=True,cycle_length=192,block_length=48,
	                )
	dataset = dataset.batch(9216, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
	dataset = dataset.prefetch(tf.data.AUTOTUNE)
	dataset = dataset.with_options(options) 

	#print(dataset)
	#for item in dataset.take(1):
	#	print(item)
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
					deterministic=False,cycle_length=8,block_length=1024,
	                )
	valiset = valiset.batch(8192, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
	valiset = valiset.prefetch(tf.data.AUTOTUNE)
	valiset = valiset.with_options(options) 
	
	checkpoint = tf.keras.callbacks.ModelCheckpoint('bact-{epoch:03d}', save_weights_only=True, save_freq='epoch')
	es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)
	
	#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), histogram_freq=1, profile_batch = '1512,2024')
	#gpus = [item.name.replace('physical_device:','').lower() for item in gpus]
	#strategy = tf.distribute.MirroredStrategy(devices=gpus)
	#with strategy.scope():
	with tf.device('/device:GPU:0'):
		model = api(None)
	model.fit(
		dataset,
		validation_data = valiset,
		epochs          = 10,
		verbose         = 1,
		callbacks       = [checkpoint, es_callback, LossHistoryCallback() ] #tensorboard_callback]
	)
