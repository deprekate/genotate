import os
import sys
import argparse
from argparse import RawTextHelpFormatter
from statistics import mode

import faulthandler

#sys.path.pop(0)
import genotate.make_train as mt
import genotate.make_model as mm
from genotate.write_genbank import Locus
#from genotate.features import Features

#from genotate.windows import get_windows
from genotate.make_train import get_windows
#from genotate.mt import get_windows

import ruptures as rpt
# TensorFlow and tf.keras
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
	tf.config.experimental.set_memory_growth(physical_devices[0], True)
	
# Helper libraries
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np


nucs = ['T', 'C', 'A', 'G']
codons = [a+b+c for a in nucs for b in nucs for c in nucs]
amino_acids = 'FFLLSSSSYY#+CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'
translate_codon = dict(zip(codons, amino_acids))
translate = lambda dna : ''.join([translate_codon[dna[i:i+3].upper()] for i in range(0,len(dna),3) ])

def predict_switches(data, min_size, pen):
	try:
		#switches = rpt.KernelCPD(kernel="linear", min_size=33).fit(strand_wise[:-3,:]).predict(pen=33)
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

def smo(data, l=10):
	out = np.zeros_like(data)
	for i in range(6):
		for j in range(3):
			out[i::6,j] = smooth_line(data[i::6,j], window_len=l)
	return out


def smooth_line(x,window_len=10,window='hamming'):
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = eval('np.'+window+'(window_len)')
    y = np.convolve(w/w.sum(),s,mode='valid')
    return y[ (window_len-1)//2 : -(window_len-1)//2 ]

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

def smoo(data, n=10):
	out = np.zeros_like(data)
	for i in range(len(data)):
		window = data[ max(i-69, i % 6) : i+80 : 6 ]
		out[i] = np.mean(window, axis=0)
	return out

def best(data):
	out = np.zeros_like(data)
	for i in range(0, len(data), 6):
		idx = np.argmax(data[i:i+6, 1])
		out[i+idx][1] = data[i+idx, 1]
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
	parser.add_argument('-g', '--genes', action="store_true")
	#parser.add_argument('-a', '--amino', action="store_true")
	parser.add_argument('-a', '--activation', action="store", default='relu', type=str, help='activation function')
	parser.add_argument('-f', '--plot_frames', action="store_true")
	parser.add_argument('-s', '--plot_strands', action="store_true")
	parser.add_argument('-t', '--trim', action="store", default=0, type=int, help='how many bases to trim off window ends')
	parser.add_argument('-r', '--reg', action="store_true", help='use kernel regularizer')
	args = parser.parse_args()

	outfile = args.outfile

	#ckpt_reader = tf.train.load_checkpoint(args.model)
	#n = len(ckpt_reader.get_tensor('layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE'))
	#model = mm.create_model_deep(n)
	model = mm.create_model_conv2(args)
	name = args.infile.split('/')[-1]
	me = name[10]
	#model.load_weights( "models/win_trim=15,reg=False,fold=" + str(me) + ",weights=1;1;1.ckpt" ).expect_partial()
	model.load_weights(args.model).expect_partial()
	#print(model.summary())
	#faulthandler.enable()

	contigs = mt.read_fasta(args.infile)
	for header in contigs:
		dna = contigs[header]
		locus = Locus(header, dna)
		generator = lambda : get_windows(dna)
		dataset = tf.data.Dataset.from_generator(
								generator,
								output_signature=(
										tf.TensorSpec(
											#shape=model.input.type_spec.shape[1:],
											shape=(1,),
											#dtype=tf.float32
											dtype=tf.string
											)
										)
								).batch(100)
		#for feature in dataset.take(1):
		#	print( feature )
		#exit()
		with tf.device('/device:CPU:0'):
			p = model.predict(dataset)
		#p = smoo(p)
		#p = best(p)
		
		if args.plot_strands:
			#p = smo(p, 30)
			#p = smoo(p)
			forward = np.array([ p[0::6,:] , p[2::6,:] , p[4::6,:] ])
			reverse = np.array([ p[1::6,:] , p[3::6,:] , p[5::6,:] ])

			strand_wise = np.array([ 
									reverse[:,:,1].sum(axis=0).clip(0,1) , 
									np.divide( reverse[:,:,2] + forward[:,:,2], 6).sum(axis=0).clip(0,1) , 
									forward[:,:,1].sum(axis=0).clip(0,1) 
									]).T
			# detection
			#algo = rpt.Pelt(model="rbf").fit(signal)
			result = rpt.KernelCPD(kernel="linear", min_size=33).fit(strand_wise).predict(pen=15)
			result = [x*3 for x in result]
			
			print("# BASE VAL1  VAL2 VAL3  VAL4")
			print("# colour 255:0:0 0:0:255 0:0:0 128:128:128 0:0:0")
			for i in range(len(strand_wise)):
				c = 1 if 3*i in result or 3*i+1 in result or 3*i+2 in result or 3*i+3 in result or 3*i+4 in result or 3*i+5 in result or 3*i+6 in result or 3*i+7 in result or 3*i+8 in result or 3*i+9 in result  else 0
				print(3*i+1, strand_wise[i,0], strand_wise[i,1], strand_wise[i,2], c, sep='\t')
				print(3*i+2, strand_wise[i,0], strand_wise[i,1], strand_wise[i,2], c, sep='\t')
				print(3*i+3, strand_wise[i,0], strand_wise[i,1], strand_wise[i,2], c, sep='\t')
			exit()
		if args.plot_frames:
			print("# BASE VAL1  VAL2 VAL3  VAL4 VAL5 VAL6 VAL7")
			print("# colour 255:0:0 0:0:255 0:0:0 255:0:255 0:128:128 128:128:128 255:195:0")
			for i in range(0,len(p), 6):
				val = []
				ig = []
				for j in range(6):
					val.append(p[i+j,1])
					ig.append(p[i+j, 2])
				v = [None] * 7
				v[0] = val[0]
				v[1] = val[2]
				v[2] = val[4]
				v[3] = val[1]
				v[4] = val[3]
				v[5] = val[5]
				v[6] = max(ig)
				print(i // 2 + 1, end='\t')
				print('\t'.join(map(str,v)))
				print(i // 2 + 2, end='\t')
				print('\t'.join(map(str,v)))
				print(i // 2 + 3, end='\t')
				print('\t'.join(map(str,v)))
			exit()
		if args.genes: # or args.amino:
			# forward[ frame : bp : type ]
			forward = np.array([ p[0::6,:] , p[2::6,:] , p[4::6,:] ])
			reverse = np.array([ p[1::6,:] , p[3::6,:] , p[5::6,:] ])
			both = np.array([ p[0::6,:] + p[1::6,:] , p[2::6,:] + p[3::6,:] , p[4::6,:] + p[5::6,:] ]).clip(0,1)
			'''
			print(forward)
			print(reverse)
			print(both)
			exit()
			print("# BASE VAL1  VAL2 VAL3 ")
			print("# colour 255:0:0 0:0:255 0:0:0")
			for n in range(both.shape[1]):
				print((3*n)+1, both[0,n,1], both[1,n,1], both[2,n,1], sep='\t')
				print((3*n)+2, both[0,n,1], both[1,n,1], both[2,n,1], sep='\t')
				print((3*n)+3, both[0,n,1], both[1,n,1], both[2,n,1], sep='\t')
			exit()
			'''

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
				index , offset , strand = max(index - 30, 0) , min(offset + 30, len(strand_wise)) , strand-1
				if strand > 0:
					locus.add_feature('mRNA', +1, [[3*index+1, 3*offset+1]] )
					for frame in [0,1,2]:
						local = forward[frame, index : offset//3*3, :]
						switches = predict_switches(local, 33, 10)
						for (left,right),label in switches.items(): 
							if label == 1:
								locus.add_feature('CDS', +1, [[3*(index+left)+frame+1, 3*(index+right)+frame]] )
				elif strand < 0:
					locus.add_feature('mRNA', -1, [[3*index+1, 3*offset+1]] )
					for frame in [0,1,2]:
						local = reverse[frame, index : offset, :]
						switches = predict_switches(local, 33, 10)
						for (left,right),label in switches: 
							if label == 1:
								locus.add_feature('CDS', -1, [[3*(index+left)+frame+1, 3*(index+right)+frame]] )

			locus.check()
			locus.write(args.outfile)
			exit()
		else:
			Y = np.argmax(p,axis=-1)
			#Y = smooth(Y)
			#Y = cutoff(Y)
			for i,row in enumerate(Y[:-4]):
				#if not i%2:
				#	print(1+i//2, p[i], p[i+1], sep='\t')
				#print(1+i//2, p[i], sep='\t')
				#continue	
				if row == 1:
					if i%2:
						print('     CDS             complement(', ((i-1)//2)+1, '..', ((i-1)//2)+3, ')', sep='')
					else:
						print('     CDS             ', (i//2)+1 , '..', (i//2)+3, sep='')
					print('                     /colour=100 100 100')
			print("//")
	
