#!/usr/bin/env python3
from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE,SIG_DFL) 

import logging
import os
import sys
import argparse
import pkg_resources
from packaging import version
import warnings
warnings.filterwarnings('ignore')

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

# backward compatibility
def sliding_window_view(ar, i):
    a = np.concatenate(( ar, ar[:-1] ))
    L = len(ar)
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, (L,L), (n,n), writeable=False)[:-i+1,:i]
if version.parse(np.__version__) < version.parse('1.20'):
    setattr(np.lib.stride_tricks, 'sliding_window_view', sliding_window_view)

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
		print("error")
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

def _print(self, item):
	if isinstance(item, str):
		self.write(item)
	else:
		self.write(str(item))

if __name__ == '__main__':
	usage = '%s [-opt1, [-opt2, ...]] infile' % __file__
	parser = argparse.ArgumentParser(description='', formatter_class=argparse.RawTextHelpFormatter, usage=usage)
	parser.add_argument('infile', type=is_valid_file, help='input file')
	parser.add_argument('-o', '--outfile', action="store", default=sys.stdout, type=argparse.FileType('w'), help='where to write output [stdout]')
	parser.add_argument('-f', '--format', help='Output the features in the specified format', type=str, default='genbank', choices=File.formats)
	#parser.add_argument('-m', '--model', help='', required=True)
	#parser.add_argument('-a', '--amino', action="store_true")
	parser.add_argument('-p', '--plot', action="store_true")
	#parser.add_argument('-s', '--size', default=30, type=int)
	#parser.add_argument('-p', '--penalty', default=10, type=int)
	args = parser.parse_args()
	args.outfile.print = _print.__get__(args.outfile)

	os.environ["CUDA_VISIBLE_DEVICES"]="0" #str(int(args.model[-2:]) % 8)
	gpus = tf.config.list_physical_devices('GPU') if hasattr(tf.config, 'list_physical_devices') else []
	for gpu in gpus:
		tf.config.experimental.set_memory_growth(gpu, True)
	
	with quiet() ,tf.device('/device:GPU:0'), quiet():
		model = [api(args)] * 5
		for i in range(5):
			path = pkg_resources.resource_filename('genotate', 'phage' + str(i))
			model[i].load_weights( path ).expect_partial()
	int8 = tf.experimental.numpy.int8 if hasattr(tf.experimental,'numpy') else np.int8	
	spec = (tf.TensorSpec(shape = (None,87), dtype = int8),
            tf.TensorSpec(shape = (None, 3), dtype = int8))

	for locus in File(args.infile):
		locus.clear()
		locus.dna = locus.dna.lower()
		locus.stops = ['taa','tga','tag']
		generator = lambda : parse_locus(locus)
		try:
			dataset = tf.data.Dataset.from_generator(
                        generator,
                        output_signature=spec
                    ) #.unbatch(),
		except:
			dataset = tf.data.Dataset.from_generator(
                        generator,
						output_types = ('int8','int8'),
						output_shapes = ( (None,87), (None,3) )
                    ) #.unbatch(),
		dataset = dataset.apply(tf.data.experimental.unbatch())
		dataset = dataset.batch(1024)

		with quiet():
			p0 = model[0].predict(dataset)
			p1 = model[1].predict(dataset)
			p2 = model[2].predict(dataset)
			p3 = model[3].predict(dataset)
			p4 = model[4].predict(dataset)
		p = np.mean([p0, p1, p2, p3, p4], axis=0)
		if args.plot:
			plot_frames(args, p)
			continue
		#p = tf.nn.softmax(p).numpy()
		#p = smoo(p)

		# create arrays of form: array[ frame : base_position : type ]
		#forward = np.array([ p[0::6,:] , p[2::6,:] , p[4::6,:] ])
		forward = np.fstack([ p[0::6,:] , p[2::6,:] , p[4::6,:] ])
		reverse = np.fstack([ p[1::6,:] , p[3::6,:] , p[5::6,:] ])
		#both = np.fstack([ p[1::6,:] , p[3::6,:] , p[5::6,:] , p[1::6,:] , p[3::6,:] , p[5::6,:] ])
		#both = np.array([ p[0::6,:] + p[1::6,:] , p[2::6,:] + p[3::6,:] , p[4::6,:] + p[5::6,:] ]).clip(0,1)
		# predict strands
		strand_wise = np.array([ 
								reverse[:,:,1].sum(axis=0).clip(0,1) , 
								np.divide( reverse[:,:,2] + forward[:,:,2], 6).sum(axis=0).clip(0,1) , 
								forward[:,:,1].sum(axis=0).clip(0,1) 
								]).T
		strand_switches = predict_switches(strand_wise, 60, 1)
		
		# predict frames of strand
		for (index,offset),strand in strand_switches.items():
			index , offset , strand = max(index - 30, 0) , min(offset + 30, len(strand_wise)-1) , strand-1
			if strand == 0:continue
			#locus.add_feature('mRNA', strand, [map(str,[3*index+1, 3*offset+1])])
			for frame in [0,1,2]:
				frame_wise = forward[frame, index : offset//3*3, :] if strand > 0 else reverse[frame, index : offset, :]
				switches = predict_switches(frame_wise, 1, 1)
				for (left,right),label in switches.items(): 
					if label == 1:
						pairs = [ list(map(str,[3*(index+left)+frame+1, 3*(index+right)+frame])) ]
						feature = locus.add_feature('CDS', strand, pairs) 
						#feature.tags['colour'] = ["100 100 100"]

		# look for stop codon readthrough
		locus.stops = locus.detect_stops()
		transl_table = 4 if 'tga' not in locus.stops else 16 if 'tag' not in locus.stops else 1
		
		#locus.write(args=args) ; exit()

		# merge regions
		locus.merge()

		#locus = dict(sorted(locus.items()))
		for key in sorted(locus):
			key.tags['transl_table'] = [transl_table]
			locus.pop(key)
			if key.partial() or key.length() >= 60:
				locus[key] = True
		
		# split regions on stop codons
		locus.split()

		# adjust ends to stop codon
		locus.adjust()

		# join partial orfs at both ends for circular genomes
		#locus.join()

		#counts = locus.count_starts()
		#print( { k: v for k, v in sorted(counts.items(), key=lambda item: item[1], reverse=True)} )
		#exit()

		locus.add_feature('source', 0, [['1',str(locus.length())]], {'stop_codon':locus.stops})
		
		# sort them so they are in numerical order instead of by frame
		# this may be a bad way to do this
		#locus = dict(sorted(locus.items()))
		for key in sorted(locus):
			locus[key] = locus.pop(key)
		try:
			locus.write(args)
		except BrokenPipeError:
			pass
	
