#!/usr/bin/env python3

import os
import sys
import argparse
from argparse import RawTextHelpFormatter
from statistics import mode

import faulthandler; faulthandler.enable()

#sys.path.pop(0)
import genotate.make_train as mt
import genotate.make_model as mm

from genotate.file import File
from genotate.make_train import get_windows
#from genotate.features import Features
#from genotate.windows import get_windows
#from genotate.mt import get_windows


	
# Helper libraries
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np
# TensorFlow and tf.keras
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
	tf.config.experimental.set_memory_growth(physical_devices[0], True)
import ruptures as rpt


nucs = ['T', 'C', 'A', 'G']
codons = [a+b+c for a in nucs for b in nucs for c in nucs]
amino_acids = 'FFLLSSSSYY#+CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'
translate_codon = dict(zip(codons, amino_acids))
translate = lambda dna : ''.join([translate_codon[dna[i:i+3].upper()] for i in range(0,len(dna),3) ])

def predict_switches(data, min_size, pen):
	try:
		switches = rpt.KernelCPD(kernel="linear", min_size=min_size).fit(data).predict(pen=pen)
	except:
		switches = [data.shape[0]]
	#return switches

	# merge adjacent regions of the same type
	merged = dict()
	for left,right in zip([0] + switches, switches): 
		label = np.argmax(data[left : right, ].mean(axis=0))
		#label = np.argmax(local[left : right, : ].mean(axis=0))
		if merged and merged[last] == label:
			del merged[last]
			last = (last[0] , right)
			merged[ last ] = label
		else:
			last = (left , right)
			merged[ last ] = label
	return merged

def is_valid_file(x):
	if not os.path.exists(x):
		raise argparse.ArgumentTypeError("{0} does not exist".format(x))
	return x


def pack(features):
	#return tf.stack(list(features.values()), axis=-1)
	a,b = tf.split(features, [3,1], axis=-1)
	return ((b,a),)



if __name__ == '__main__':
	usage = '%s [-opt1, [-opt2, ...]] infile' % __file__
	parser = argparse.ArgumentParser(description='', formatter_class=RawTextHelpFormatter, usage=usage)
	parser.add_argument('infile', type=is_valid_file, help='input file')
	parser.add_argument('-o', '--outfile', action="store", default=sys.stdout, type=argparse.FileType('w'), help='where to write output [stdout]')
	parser.add_argument('-m', '--model', help='', required=True)
	parser.add_argument('-c', '--cutoff', help='The minimum cutoff length for runs', type=int, default=29)
	parser.add_argument('-g', '--genes', action="store_true")
	#parser.add_argument('-a', '--amino', action="store_true")
	parser.add_argument('-a', '--activation', action="store", default='relu', type=str, help='activation function')
	parser.add_argument('-f', '--plot_frames', action="store_true")
	parser.add_argument('-s', '--plot_strands', action="store_true")
	parser.add_argument('-t', '--trim', action="store", default=15, type=int, help='how many bases to trim off window ends')
	parser.add_argument('-r', '--reg', action="store_true", help='use kernel regularizer')
	args = parser.parse_args()

	outfile = args.outfile

	#ckpt_reader = tf.train.load_checkpoint(args.model)
	#n = len(ckpt_reader.get_tensor('layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE'))
	#model = mm.create_model_deep(n)
	model = mm.create_model_blend(args)
	name = args.infile.split('/')[-1]
	me = name[10] if len(name) > 10 else None
	#model.load_weights( "out/win_sub_w117_kern3/win_sub_trim=15,reg=False,fold=" + str(me) + ".ckpt" ).expect_partial()
	model.load_weights(args.model).expect_partial()
	#print(model.summary())
	#faulthandler.enable()

	#contigs = mt.read_fasta(args.infile)
	file = File(args.infile)
	for locus in file:
		dna = locus.seq()
		#for k,v in locus.slippage_sites.items():
		#	print(k,v)
		#exit()
		generator = lambda : get_windows(dna)
		dataset = tf.data.Dataset.from_generator(
								generator,
								output_signature=(
										tf.TensorSpec(
											#shape=model.input.type_spec.shape[1:],
											shape=(4,),
											#dtype=tf.float32
											dtype=tf.string
											)
										)
								).batch(10)
		tdata = dataset.map(pack)
		#for feature in dataset.take(1):
		'''
		for feature in tdata:
			print( feature )
			exit()
		exit()
		'''
		with tf.device('/device:CPU:0'):
			p = model.predict(tdata)

		if args.plot_frames:
			import genotate.file_handling as fh
			fh.plot_frames(p)
			exit()
		#p = tf.nn.softmax(p).numpy()
		#p = smoo(p)
		#p = best(p)
		
		# create arrays of form: array[ frame : bp : type ]
		forward = np.array([ p[0::6,:] , p[2::6,:] , p[4::6,:] ])
		reverse = np.array([ p[1::6,:] , p[3::6,:] , p[5::6,:] ])
		#both = np.array([ p[0::6,:] + p[1::6,:] , p[2::6,:] + p[3::6,:] , p[4::6,:] + p[5::6,:] ]).clip(0,1)

		# predict strands
		strand_wise = np.array([ 
								reverse[:,:,1].sum(axis=0).clip(0,1) , 
								np.divide( reverse[:,:,2] + forward[:,:,2], 6).sum(axis=0).clip(0,1) , 
								forward[:,:,1].sum(axis=0).clip(0,1) 
								]).T
		#forward[:,:,1] = forward[:,:,1] + reverse[:,:,1]
		#reverse[:,:,1] = forward[:,:,1] + reverse[:,:,1]
	
		strand_switches = predict_switches(strand_wise, 33, 33)

		# predict frames of strand
		for (index,offset),strand in strand_switches.items():
			index , offset , strand = max(index - 30, 0) , min(offset + 30, len(strand_wise)-1) , strand-1
			if strand > 0:
				locus.add_feature('mRNA', +1, [[str(3*index+1), str(3*offset+1)]] )
				for frame in [0,1,2]:
					local = forward[frame, index : offset//3*3, :]
					switches = predict_switches(local, 33, 10)
					for (left,right),label in switches.items(): 
						if label == 1:
							locus.add_feature('CDS', +1, [[str(3*(index+left)+frame+1), str(3*(index+right)+frame-2)]] )
						#elif label == 2:
						#	locus.add_feature('misc_feature', +1, [[3*(index+left)+frame+1, 3*(index+right)+frame]] )
			elif strand < 0:
				locus.add_feature('mRNA', -1, [[3*index+1, 3*offset+1]] )
				for frame in [0,1,2]:
					local = reverse[frame, index : offset, :]
					switches = predict_switches(local, 33, 10)
					for (left,right),label in switches.items(): 
						if label == 1:
							locus.add_feature('CDS', -1, [[str(3*(index+left)+frame+1), str(3*(index+right)+frame-2)]] )
						#elif label == 2:
						#	locus.add_feature('misc_feature', -1, [[3*(index+left)+frame+1, 3*(index+right)+frame]] )

		locus.merge()
		#locus.rbs()
		#locus.mfe()
		#locus.rare_codons()
		locus.write(args.outfile)
		exit()
	
