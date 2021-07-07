import os
import sys
import argparse
from argparse import RawTextHelpFormatter
from statistics import mode

import faulthandler

from find_frameshifts import find_frameshifts
#sys.path.pop(0)
import genotate.make_train as mt
import genotate.make_model as mm
from genotate.features import Features

#from genotate.windows import get_windows
from genotate.make_train import get_windows

# TensorFlow and tf.keras
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# Helper libraries
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np

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
			#window = var[ j , max(i-1, 0) : i+2 ]
			out[6*i+j] = mode(window)
	return out

def cutoff(data, c=29):
	out = np.zeros_like(data)
	data[6] = 1
	data[12] = 1
	var = np.array([
					data[0::6], 
					data[1::6],
					data[2::6],
					data[3::6],
					data[4::6],
					data[5::6]
					])
	for i in range(var.shape[1]):
		for j in range(6):
			if var[j,i] == 1:
				befor = np.flip( var[ j , max(i-39, 0) : i+1  ] )
				after =          var[ j ,     i        : i+40 ]
				b = np.where(np.append(befor, 0) != 1)[0][0]
				a = np.where(np.append(after, 0) != 1)[0][0]
				if b+a > c:
					out[6*i+j] = 1
				else:
					out[6*i+j] = 0
			else:
				out[6*i+j] = var[j,i]
	return out

def wrapper(thing):
	for item in get_windows(thing):
		yield item


if __name__ == '__main__':
	usage = '%s [-opt1, [-opt2, ...]] infile' % __file__
	parser = argparse.ArgumentParser(description='', formatter_class=RawTextHelpFormatter, usage=usage)
	parser.add_argument('infile', type=is_valid_file, help='input file')
	parser.add_argument('-o', '--outfile', action="store", default=sys.stdout, type=argparse.FileType('w'), help='where to write output [stdout]')
	parser.add_argument('-m', '--model', help='', required=True)
	parser.add_argument('-c', '--cutoff', help='The minimum cutoff length for runs', type=int, default=29)
	args = parser.parse_args()
	'''
	if args.labels: print("\t".join(['ID','TYPE','GC'] + translate.amino_acids))
		exit()
	'''


	ckpt_reader = tf.train.load_checkpoint(args.model)
	model = mm.create_model(len(ckpt_reader.get_tensor('layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE')))
	model.load_weights(args.model).expect_partial()

	#faulthandler.enable()
	contigs = mt.read_fasta(args.infile)
	for header in contigs:
		dna = contigs[header]
		generator = lambda : get_windows(dna)
		dataset = tf.data.Dataset.from_generator(
								generator,
								output_signature=(
										tf.TensorSpec(
											shape=model.input.type_spec.shape[1:],
											dtype=tf.float32
											)
										)
								).batch(32)
		#for feature in dataset.take(1):
		#	print( feature )
		#exit()
		p = model.predict(dataset)
		if p.shape[1] == 1:
			Y = np.round(p.flatten())
		else:
			Y = np.argmax(p,axis=-1)

		#contig_features = Features(**vars(args))
		#contig_features.parse_contig(header, contigs[header], Y)
		#for orfs in contig_features.iter_orfs('longest'):
		#	for orf in orfs:
		#		print(orf)

		#find_frameshifts(dna, p)
		#exit()
		
		#Y = smooth(Y)
		#Y = cutoff(Y)
		for i,row in enumerate(Y[:-4]):
			#if not i%2:
				#print(1+i//2, p[i], p[i+1])
			#print(1+i//2, p[i])
			#continue	
			if row == 1:
				if i%2:
					print('     CDS             complement(', ((i-1)//2)+1, '..', ((i-1)//2)+3, ')', sep='')
				else:
					print('     CDS             ', (i//2)+1 , '..', (i//2)+3, sep='')
				print('                     /colour=100 100 100')

		print("//")

