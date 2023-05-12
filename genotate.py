#!/usr/bin/env python3
from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE,SIG_DFL) 

import logging
import os
import sys
import argparse
from argparse import RawTextHelpFormatter
from statistics import mode

import faulthandler

#os.environ["CUDA_VISIBLE_DEVICES"]="0"str(int(args.model[-2:]) % 8)
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
sys.path.pop(0)
from genotate.file import File
import genotate.make_train as mt
from genotate.make_model import create_model_blend, blend, api
from genotate.make_train import get_windows
from genotate.functions import *
	
# Helper libraries
import numpy as np
# TensorFlow and tf.keras
import tensorflow as tf
#gpus = tf.config.list_physical_devices('GPU')
#for gpu in gpus:
#    tf.config.experimental.set_memory_growth(gpu, True)
import ruptures as rpt


def fstack(data, axis=0):
	shape = min([item.shape for item in data])
	return np.stack([np.resize(item, shape) for item in data], axis)
setattr(np, 'fstack', fstack)

class quiet:
	def __enter__(self):
		fd = os.open('/dev/null',os.O_WRONLY)
		self.savefd = os.dup(1)
		os.dup2(fd,1)
	def __exit__(self, *args):
		os.dup2(self.savefd,1)

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


if __name__ == '__main__':
	usage = '%s [-opt1, [-opt2, ...]] infile' % __file__
	parser = argparse.ArgumentParser(description='', formatter_class=RawTextHelpFormatter, usage=usage)
	parser.add_argument('infile', type=is_valid_file, help='input file')
	parser.add_argument('-o', '--outfile', action="store", default=sys.stdout, type=argparse.FileType('w'), help='where to write output [stdout]')
	parser.add_argument('-f', '--format', help='Output the features in the specified format', type=str, default='genbank', choices=['gbk','gff3','gff'])
	parser.add_argument('-m', '--model', help='', required=True)
	#parser.add_argument('-a', '--amino', action="store_true")
	parser.add_argument('-g', '--graph', action="store_true")
	parser.add_argument('-s', '--size', default=30, type=int)
	parser.add_argument('-p', '--penalty', default=10, type=int)
	args = parser.parse_args()

	os.environ["CUDA_VISIBLE_DEVICES"]="0" #str(int(args.model[-2:]) % 8)
	gpus = tf.config.list_physical_devices('GPU')
	for gpu in gpus:
		tf.config.experimental.set_memory_growth(gpu, True)
	#ckpt_reader = tf.train.load_checkpoint(args.model)
	#n = len(ckpt_reader.get_tensor('layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE'))
	#model = mm.create_model_deep(n)
	#model = blend(args)
	k = 1 #int(os.path.basename(args.infile)[11]) % 5
	args.model = args.model.replace('#', str(k))
	with quiet() ,tf.device('/device:GPU:0'), quiet():
		model = api(args)
		#name = args.infile.split('/')[-1]
		#me = name[10] if len(name) > 10 else None
		#model.load_weights( "out/win_sub_w117_kern3/win_sub_trim=15,reg=False,fold=" + str(me) + ".ckpt" ).expect_partial()
		model.load_weights(args.model).expect_partial()
		#print(model.summary())
		#faulthandler.enable()
	spec = (tf.TensorSpec(shape = (None,87), dtype = tf.experimental.numpy.int8),
            tf.TensorSpec(shape = (None, 3), dtype = tf.experimental.numpy.int8))

	genbank = File(args.infile)
	for locus in genbank:
		locus.clear()
		locus.stops = ['taa','tga','tag']
		generator = lambda : parse_locus(locus)
		dataset = tf.data.Dataset.from_generator(
                        generator,
                        output_signature=spec
                    ) #.unbatch(),
		dataset = dataset.apply(tf.data.experimental.unbatch())
		dataset = dataset.batch(1024)
		#for feature in dataset.take(1):
		#	print( feature )
		#	exit()
		'''
		for block,label in parse_locus(locus):
			for i in range(len(block)):
				row = block[i,:]
				dna = to_dna(row)
				#print(i//2,dna)
				row = tf.expand_dims(row, axis=0)
				#print(row)
				p = model.predict(row, verbose=0)
				print(i//2, dna, p)
		exit()
		'''
		with quiet():
			p = model.predict(dataset)
	
		if args.graph:
			plot_frames(p)
			continue
		#p = tf.nn.softmax(p).numpy()
		#p = smoo(p)
		#p = best(p)

		# create arrays of form: array[ frame : base_position : type ]
		#forward = np.array([ p[0::6,:] , p[2::6,:] , p[4::6,:] ])
		forward = np.fstack([ p[0::6,:] , p[2::6,:] , p[4::6,:] ])
		reverse = np.fstack([ p[1::6,:] , p[3::6,:] , p[5::6,:] ])
		#both = np.array([ p[0::6,:] + p[1::6,:] , p[2::6,:] + p[3::6,:] , p[4::6,:] + p[5::6,:] ]).clip(0,1)
		# predict strands
		strand_wise = np.array([ 
								reverse[:,:,1].sum(axis=0).clip(0,1) , 
								np.divide( reverse[:,:,2] + forward[:,:,2], 6).sum(axis=0).clip(0,1) , 
								forward[:,:,1].sum(axis=0).clip(0,1) 
								]).T
		strand_switches = predict_switches(strand_wise, 33, 33)
		
		# predict frames of strand
		for (index,offset),strand in strand_switches.items():
			index , offset , strand = max(index - 30, 0) , min(offset + 30, len(strand_wise)-1) , strand-1
			#locus.add_feature('mRNA', strand, [map(str,[3*index+1, 3*offset+1])])
			if strand == 0:continue
			for frame in [0,1,2]:
				frame_wise = forward[frame, index : offset//3*3, :] if strand > 0 else reverse[frame, index : offset, :]
				switches = predict_switches(frame_wise, args.size, args.penalty)
				for (left,right),label in switches.items(): 
					if label == 1:
						pairs = [ list(map(str,[3*(index+left)+frame+1, 3*(index+right)+frame])) ]
						feature = locus.add_feature('CDS', strand, pairs) 
						feature.tags['colour'] = ["100 100 100"]

		# look for stop codon readthrough
		locus.stops = locus.detect_stops()
		#locus.write(open('before.gb','w'), args=args)

		# merge regions
		locus.merge()

		for key in sorted(locus):
			locus[key] = locus.pop(key)
		
		# split regions on stop codons
		locus.split()

		# adjust ends
		locus.adjust()

		# join partial orfs at both ends
		locus.join()

		#counts = locus.count_starts()
		#print( { k: v for k, v in sorted(counts.items(), key=lambda item: item[1], reverse=True)} )
		#exit()

		# sort them so they are in numerical order instead of by frame
		# this may be a bad way to do this
		for key in sorted(locus):
			locus[key] = locus.pop(key)
		try:
			locus.write(args.outfile, args=args)
		except BrokenPipeError:
			pass
	
