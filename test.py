import os
import sys
import time

os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from genotate.fun import GenomeDataset, GenDataset, parse_genbank
from genotate.make_model import create_model_blend, blend, api
import tensorflow as tf
import numpy as np
from genbank.file import File
from genotate.functions import parse_locus, to_dna
import datetime
import time


os.environ["CUDA_VISIBLE_DEVICES"]="3" #,4,5,7"
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

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


def pack(features): #labels, windows): #labels,datas,windows): #features):
	#labels,windows,datas = tf.split(features, [1,99,3], axis=-1)
	labels,windows = tf.split(features, [1,99], axis=-1)
	#windows = tf.unstack(windows)
	#labels = tf.cast(labels, dtype=tf.int32)
	labels = tf.one_hot(labels, depth=3, axis=-1) #, dtype=tf.int32)
	labels = tf.squeeze(labels)
	#labels = tf.reshape(labels, [-1])
	#labels = tf.unstack(labels)
	#print(labels) ; print(datas) ; print(windows)
	#return (windows, datas) , labels
	return windows , labels

def iter_genbank(infile):
	genbank = File(infile.decode())
	for locus in genbank:
		for data in parse_locus(locus):
			yield data

'''
dataset = GenDataset("/home/mcnair/assembly/phages/train/GCA_000851005.1.gbff.gz")
for window,label in dataset.take(50000):
	print(label, to_dna(window.numpy()))
exit()
'''

spec = (tf.TensorSpec(shape = (None,99), dtype = tf.int32),tf.TensorSpec(shape = (None,3), dtype = tf.int32))
directory = sys.argv[1]
filenames = [os.path.join(directory,f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
filenames = filenames[:1000]
#print(filenames)
dataset = tf.data.Dataset.from_tensor_slices(filenames)
dataset = dataset.interleave(
                #lambda x: GenDataset(x), #.flat_map(tf.data.Dataset.from_tensor_slices),
				lambda x: GenDataset(x).flat_map(lambda *x : tf.data.Dataset.from_tensor_slices(x)),
				#lambda x: tf.data.Dataset.from_generator(
				#		parse_genbank,
				#		args=(x,),
				#		output_signature=spec,
				#),
				deterministic=False,
                num_parallel_calls=16, #tf.data.AUTOTUNE,
                #cycle_length=128,
                block_length=128,
                )
#dataset = dataset.unbatch()
#dataset = dataset.cache('cache')
#dataset = dataset.map(pack, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
#dataset = dataset.map(pack, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
dataset = dataset.batch(4096, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
dataset = dataset.prefetch(tf.data.AUTOTUNE)
#print(dataset)
#for item in dataset.take(1):
#	print(item)
#exit()

#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), histogram_freq=1, profile_batch = '1512,2024')
#gpus = [item.name.replace('physical_device:','').lower() for item in gpus]
#strategy = tf.distribute.MirroredStrategy(devices=gpus)
#with strategy.scope():
with tf.device('/device:GPU:0'):
	model = api(None)
model.fit(
	dataset,
	#validation_data = valset,
	epochs          = 3,
	verbose         = 1,
	#callbacks       = [tensorboard_callback]
)
#benchmark(dataset)
