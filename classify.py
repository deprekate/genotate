import os
import sys
import argparse
from argparse import RawTextHelpFormatter
from statistics import mode

import make_train as mt
import make_model as mm

# TensorFlow and tf.keras
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# Helper libraries
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def is_valid_file(x):
	if not os.path.exists(x):
		raise argparse.ArgumentTypeError("{0} does not exist".format(x))
	return x

def mode(a, axis=0):
	scores = np.unique(np.ravel(a))       # get ALL unique values
	testshape = list(a.shape)
	testshape[axis] = 1
	oldmostfreq = np.zeros(testshape)
	oldcounts = np.zeros(testshape)
	for score in scores:
		template = (a == score)
		counts = np.expand_dims(np.sum(template, axis),axis)
		mostfrequent = np.where(counts > oldcounts, score, oldmostfreq)
		oldcounts = np.maximum(counts, oldcounts)
		oldmostfreq = mostfrequent
	return mostfrequent[0] #, oldcounts

def pack(features, label):
  return tf.stack(list(features.values()), axis=-1), label

def smooth(data):
	out = np.zeros_like(data)
	var = np.array([
					data[0::6], 
					data[1::6],
					data[2::6],
					data[3::6],
					data[4::6],
					data[5::6]
					])
	for i in range(var.shape[1]):
		#counts = np.count_nonzero(var[:,max(i-19, 0) : i+20] == 2, axis=1)
		#idx = np.argmax(counts)
		#if counts[idx] >= 3:
		#	out[6*i+idx] = 2
		#out[i:i+5] = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr = var[ : , max(i-19, 0) : i+20 ] )
		for j in range(6):
			window = var[ j , max(i-19, 0) : i+20 ]
			out[6*i+j] = mode(window)
	return out


if __name__ == '__main__':
	usage = '%s [-opt1, [-opt2, ...]] infile' % __file__
	parser = argparse.ArgumentParser(description='', formatter_class=RawTextHelpFormatter, usage=usage)
	parser.add_argument('infile', type=is_valid_file, help='input file')
	parser.add_argument('-o', '--outfile', action="store", default=sys.stdout, type=argparse.FileType('w'), help='where to write output [stdout]')
	parser.add_argument('-m', '--model', help='')
	args = parser.parse_args()
	'''
	if args.labels: print("\t".join(['ID','TYPE','GC'] + translate.amino_acids))
		exit()
	'''
	
	ckpt_reader = tf.train.load_checkpoint(args.model)
	model = mm.create_model3(len(ckpt_reader.get_tensor('layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE')))
	model.load_weights(args.model)
	
	contigs = mt.read_fasta(args.infile)
	for header in contigs:
		dataset = tf.data.Dataset.from_generator(
								mt.get_windows,
								args=[contigs[header]],
								#output_shapes = (121,), output_types=tf.float32,
								#output_signature=(tf.TensorSpec(shape=(121,), dtype=tf.float32))
								output_signature=(
										tf.TensorSpec(
											shape=model.input.type_spec.shape[1:],
											dtype=tf.float32
											)
										)
								).batch(1)
		#for feature in dataset.take(1):
		#	print( feature )
		#exit()
	
		p = model.predict(dataset)
		#Y = np.round(p.flatten())
		Y = np.argmax(p,axis=-1)
		Y = smooth(Y)

		for i,row in enumerate(Y):
			#if not i%2:
			#	print(1+i//2, p[i], p[i+1])
			if row == 1:
				if i%2:
					print('     CDS             complement(', ((i-1)//2)+1, '..', ((i-1)//2)+3, ')', sep='')
				else:
					print('     CDS             ', (i//2)+1 , '..', (i//2)+3, sep='')
				print('                     /colour=100 100 100')
			'''
			#elif row == 2:
			#	print('     gap             ', (i//2)+1 , '..', (i//2)+3, sep='')
			'''
			'''
			if row == 2:
				print('     CDS             complement(', i+1, '..', i+3, ')', sep='')
				print('                     /colour=100 100 100')
			elif row == 1:
				print('     CDS             ', i+1 , '..', i+3, sep='')
				print('                     /colour=100 100 100')
			'''



