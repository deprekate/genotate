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
		counts = np.count_nonzero(var[:,max(i-19, 0) : i+20] == 2, axis=1)
		idx = np.argmax(counts)
		if counts[idx] >= 3:
			out[6*i+idx] = 2
	return out


if __name__ == '__main__':
	usage = '%s [-opt1, [-opt2, ...]] infile' % __file__
	parser = argparse.ArgumentParser(description='', formatter_class=RawTextHelpFormatter, usage=usage)
	parser.add_argument('infile', type=is_valid_file, help='input file')
	parser.add_argument('-o', '--outfile', action="store", default=sys.stdout, type=argparse.FileType('w'), help='where to write output [stdout]')
	parser.add_argument('-m', '--model', required=True, help='')
	args = parser.parse_args()
	'''
	if args.labels: print("\t".join(['ID','TYPE','GC'] + translate.amino_acids))
		exit()
	'''
	
	ckpt_reader = tf.train.load_checkpoint(args.model)
	model = mm.create_model5(len(ckpt_reader.get_tensor('layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE')))
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
		#Y = np.argmax(p,axis=-1)
		#Y = smooth(Y)

		TP = FP = TN = FN = 0
		i = 1
		for [Yhat],[Y,*_] in zip(np.round(p), mt.read_genbank(args.infile. replace('fna', 'gbk'))):
			#print(Yhat, Y)
			if Y:
				if Yhat:
					TP += 1
				else:
					FN += 1
					print(i//2 + 1, Y, Yhat)
			else:
				if Yhat:
					FP += 1
				else:
					TN += 1
			i += 1

		print(TP, FP, TN, FN, sep='\t')
		PRECIS = TP / (TP + FP)
		RECALL = TP / (TP + FN)
		ACCURA = (TP + TN) / (TP + FN + TN + FP)
		print(PRECIS, 'PRECIS', sep='\t')
		print(RECALL, 'RECALL', sep='\t')
		print(ACCURA, 'ACCURA', sep='\t')
		F1 = 2 * PRECIS*RECALL / (PRECIS+RECALL) if (PRECIS+RECALL) else 0
		print(F1 ,'F1', sep='\t')


