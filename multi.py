import os
import sys
import re
import argparse
from argparse import RawTextHelpFormatter
from os import listdir
from os.path import isfile, join
import datetime

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
import tensorflow as tf
#tf.keras.backend.set_floatx('float16')
import numpy as np


from genbank.file import File
#from genotate.make_train import get_windows, parse_genbank
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

def skew(seq, nucs):
	self = lambda : none
	self.sequence = seq
	self.windowsize = self.stepsize = 100 #int(len(self.sequence) / 1000)
	(self.nuc_1,self.nuc_2) = nucs
	
	cumulative = 0
	cm_list = []
	i = int(self.windowsize / 2)
	for each in range(len(self.sequence) // self.stepsize):
		if i < len(self.sequence):
			a = self.sequence[i - int(self.windowsize / 2):i + int(self.windowsize / 2)].count(self.nuc_1)
			b = self.sequence[i - int(self.windowsize / 2):i + int(self.windowsize / 2)].count(self.nuc_2)
			s = (a - b) / (a + b) if (a + b) else 0
			cumulative = cumulative + s
			cm_list.append(cumulative)
			i = i + self.stepsize
	slopes = []
	for i in range(len(cm_list)):
		win = cm_list[max(i-5,0):i+5]
		m,b = np.polyfit(list(range(len(win))),win, 1)
		slopes.append(m)
	slopes.append(m)
	return slopes

def rev_comp(seq):
	seq_dict = {'a':'t','t':'a','g':'c','c':'g',
                'n':'n',
                'r':'y','y':'r','s':'s','w':'w','k':'m','m':'k',
                'b':'v','v':'b','d':'h','h':'d'}
	return "".join([seq_dict[base] for base in reversed(seq)])

def parse_genbank(infile):
	genbank = File(infile.decode())
	# label the positions
	for locus in genbank:
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
		gc = locus.gc_content() 
		at_skew = skew(dna, 'at')
		gc_skew = skew(dna, 'gc')
		#forward = 'n'*48 +          dna  + 'n'*50
		#reverse = 'n'*48 + rev_comp(dna) + 'n'*50
		forward = [0]*48 + [0]*len(dna) + [0]*50
		reverse = [0]*48 + [0]*len(dna) + [0]*50
		for i,base in enumerate(dna):
			if base in 'acgt':
				forward[i+48] = ((ord(base) >> 1) & 3) + 1
				reverse[i+48] = ((forward[i+48] - 3) % 4) + 1
		#a = np.zeros([2*len(dna)-2, 103])
		a = np.zeros([6, 103])
		for n in range(0, len(dna)-2, 3):
			for f in [0,1,2]:
				#yield positions.get( n+f, 2) , [ gc,  at_skew[n//100],  gc_skew[n//100] ] , forward[n+f : n+f+99 ]
				#yield positions.get(-n+f, 2) , [ gc, -at_skew[n//100], -gc_skew[n//100] ] , reverse[n+f : n+f+99 ][::-1]
				a[f  ,0] = positions.get( n+f, 2)
				a[f+1,0] = positions.get(-n+f, 2)
				a[f  ,1] = gc
				a[f+1,1] = gc
				a[f  ,2] =  at_skew[n//100]
				a[f+1,2] = -at_skew[n//100]
				a[f  ,3] =  gc_skew[n//100]
				a[f+1,3] =  gc_skew[n//100]
				a[f  ,4:103] = forward[n+f : n+f+99 ]
				a[f+1,4:103] = reverse[n+f : n+f+99 ][::-1]
			yield a

def make_dataset(filename):
	ds = tf.data.Dataset.from_generator(
		lambda: parse_genbank(filename), output_types=tf.float32
	)
	it = ds.__iter__()
	return it

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

	with tf.device('/device:GPU:0'):
		#datasets = [make_dataset(i) for i in filenames]
		#dataset = tf.data.Dataset.from_tensor_slices(datasets)
		#dataset = dataset.interleave(lambda x: x, cycle_length=tf.data.AUTOTUNE)
		#for feature in dataset.take(1):
		#	print( feature )
		#exit()

		dataset = tf.data.Dataset.from_tensor_slices(filenames)
		dataset = dataset.interleave(
							lambda x: tf.data.Dataset.from_generator(
								parse_genbank,
								args=(x,),
								output_signature=(
									tf.TensorSpec(shape=(6,103),dtype=tf.float32)
									#(
									#tf.TensorSpec(shape=(5,),dtype=tf.string)
									#tf.TensorSpec(shape=(),dtype=tf.int32),
									#tf.TensorSpec(shape=(3,),dtype=tf.float32),
									#tf.TensorSpec(shape=(99,),dtype=tf.int32)
									#tf.TensorSpec(shape=(),dtype=tf.string)
									#)
								)
							),
							num_parallel_calls=tf.data.AUTOTUNE,
							#cycle_length=70,
							block_length=100
							)
		#print(dataset)
		dataset = dataset.unbatch()
		dataset = dataset.map(pack)
		#dataset = dataset.shuffle(buffer_size=1000)
		dataset = dataset.batch(4000)
		dataset = dataset.prefetch(tf.data.AUTOTUNE)

		log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
		tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch = '5,25')

		#for feature in dataset.take(1):
		#	print( feature )
		#exit()

		model = blend(args)
		#model = create_model_blend(args)
		model.fit(
			dataset,
			epochs          = 3,
			verbose         = 1,
			callbacks=[tensorboard_callback]
		)
