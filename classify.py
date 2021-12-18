import os
import sys
import argparse
from argparse import RawTextHelpFormatter
from statistics import mode

import faulthandler

#sys.path.pop(0)
import genotate.make_train as mt
import genotate.make_model as mm
import genotate.codons as cd
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
	parser.add_argument('-a', '--amino', action="store_true")
	parser.add_argument('-f', '--plot_frames', action="store_true")
	parser.add_argument('-s', '--plot_strands', action="store_true")
	args = parser.parse_args()
	'''
	if args.labels: print("\t".join(['ID','TYPE','GC'] + translate.amino_acids))
		exit()
	'''

	outfile = args.outfile

<<<<<<< HEAD
	#ckpt_reader = tf.train.load_checkpoint(args.model)
	#n = len(ckpt_reader.get_tensor('layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE'))
	#model = mm.create_model_deep(n)
	model = mm.create_model_conv()
	n = 1
=======
	ckpt_reader = tf.train.load_checkpoint(args.model)
	n = len(ckpt_reader.get_tensor('layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE'))
	model = mm.create_model(n)
>>>>>>> 8baae49aed44c59f19e1c6e36492b2e1c6dd1b28
	model.load_weights(args.model).expect_partial()
	#print(model.summary())
	#faulthandler.enable()

	contigs = mt.read_fasta(args.infile)
	for header in contigs:
		dna = contigs[header]
		locations = cd.Locations(dna)
		generator = lambda : get_windows(dna)
		dataset = tf.data.Dataset.from_generator(
								generator,
								output_signature=(
										tf.TensorSpec(
											#shape=model.input.type_spec.shape[1:],
											shape=(n,),
<<<<<<< HEAD
											#dtype=tf.float32
											dtype=tf.string
=======
											dtype=tf.float32
>>>>>>> 8baae49aed44c59f19e1c6e36492b2e1c6dd1b28
											)
										)
								).batch(32)
		#for feature in dataset.take(1):
		#	print( feature )
		#exit()
		with tf.device('/device:CPU:0'):
			p = model.predict(dataset)
		#p = smoo(p)
		#p = best(p)
		'''
		if True:
			for i in range(0,len(p), 6):
				for f in range(6):
					print('\t'.join([str(item) for item in p[i+f]]), end='\t')
				print()
		exit()
		'''
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
			'''
			f = p[0::6,1] + p[2::6,1] + p[4::6,1]
			#f = smooth_line(f.flatten(),30)
			#f = smoo(f.flatten(),30)
			r = p[1::6,1] + p[3::6,1] + p[5::6,1]
			#r = smooth_line(r.flatten(), 30)
			#r = smoo(r.flatten(), 30)
			ig = (p[0::6,2] + p[2::6,2] + p[4::6,2] + p[1::6,2] + p[3::6,2] + p[5::6,2]) / 6
			#ig = smooth_line(ig.flatten(), 30)
			#ig = smoo(ig.flatten(), 30)
			signal = np.array([f.clip(0.1,0.9), r.clip(0.1,0.9), ig/6]).T
			'''
			
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
			print("# BASE VAL1  VAL2 VAL3  VAL4 VAL5 VAL6")
			print("# colour 255:0:0 0:0:255 0:0:0 255:0:255 0:128:128 128:128:128")
			for i in range(0,len(p), 6):
				val = []
				for j in range(6):
					val.append(p[i+j,1])
				v = [None] * 6
				v[0] = val[0]
				v[1] = val[2]
				v[2] = val[4]
				v[3] = val[1]
				v[4] = val[3]
				v[5] = val[5]
				print(i // 2 + 1, end='\t')
				print('\t'.join(map(str,v)))
				print(i // 2 + 2, end='\t')
				print('\t'.join(map(str,v)))
				print(i // 2 + 3, end='\t')
				print('\t'.join(map(str,v)))
			exit()
		if args.genes or args.amino:
			if not args.amino:
				outfile.write('LOCUS       ')
				outfile.write(header)
				outfile.write(str(len(dna)).rjust(10))
				outfile.write(' bp    DNA             UNK')
				outfile.write('\n')
				outfile.write('DEFINITION  ' + header + '\n')
				outfile.write('FEATURES             Location/Qualifiers\n')
				outfile.write('     source          1..')
				outfile.write(str(len(dna)))
				outfile.write('\n')
		
			# forward[ frame : bp : type ]
			forward = np.array([ p[0::6,:] , p[2::6,:] , p[4::6,:] ])
			reverse = np.array([ p[1::6,:] , p[3::6,:] , p[5::6,:] ])
			'''
			print("# BASE VAL1  VAL2 VAL3 ")
			print("# colour 255:0:0 0:0:255 0:0:0")
			for n,row in enumerate(forward.mean(axis=0)):
				print((3*n)+1, row[0], row[1], row[2], sep='\t')
				print((3*n)+2, row[0], row[1], row[2], sep='\t')
				print((3*n)+3, row[0], row[1], row[2], sep='\t')
			'''
			# detection
			strand_wise = np.array([ 
									reverse[:,:,1].sum(axis=0).clip(0,1) , 
									np.divide( reverse[:,:,2] + forward[:,:,2], 6).sum(axis=0).clip(0,1) , 
									forward[:,:,1].sum(axis=0).clip(0,1) 
									]).T
			strand_result = rpt.KernelCPD(kernel="linear", min_size=33).fit(strand_wise[:-3,:]).predict(pen=33)

			last = 0
			for curr in strand_result: 
				strand = np.argmax(strand_wise[last : curr, ].mean(axis=0)) - 1
				#print("mrna", last*3, curr*3, strand, sep='\t')
				# forward
				if strand > 0:
					for frame in [0,1,2]:
						local = forward[frame, last : curr, :]
						result_frame = rpt.KernelCPD(kernel="linear", min_size=33).fit(local).predict(pen=20)
						left = 0
						for right in result_frame:
							label = np.argmax(local[left : right, : ].mean(axis=0))
							if label == 1:
								print('     CDS             ', 3*(last+left)+frame+1, "..", 3*(last+right)+frame+1, sep='')
								print('                     /colour=100 100 100')
							left = right
						
					pass
					'''
					outfile.write("     mRNA            " + str(left) + ".." + str(right))
					outfile.write('\n')
					outfile.write("                     /colour=200 0 0\n" )
					'''
				# reverse
				elif strand < 0:
					for frame in [0,1,2]:
						local = reverse[frame, last : curr, :]
						result_frame = rpt.KernelCPD(kernel="linear", min_size=33).fit(local).predict(pen=20)
						left = 0
						for right in result_frame:
							label = np.argmax(local[left : right, : ].mean(axis=0))
							if label == 1:
								print('     CDS             complement(', 3*(last+left)+frame, "..", 3*(last+right)+frame, ")", sep='')
								print('                     /colour=100 100 100')
							left = right
					pass
					'''
					outfile.write("     mRNA            complement(" + str(left) + ".." + str(right) + ")" )
					outfile.write('\n')
					outfile.write("                     /colour=200 0 0\n" )
					'''
				last = curr
		else:
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
				#	print(1+i//2, p[i], p[i+1], sep='\t')
				#print(1+i//2, p[i], sep='\t')
				#continue	
				if row == 1:
					if i%2:
						print('     CDS             complement(', ((i-1)//2)+1, '..', ((i-1)//2)+3, ')', sep='')
					else:
						print('     CDS             ', (i//2)+1 , '..', (i//2)+3, sep='')
					print('                     /colour=100 100 100')
				# BOTH
				'''
				if row == 1:
					print('     CDS             ', i+1 , '..', i+3, sep='')
					print('                     /colour=100 100 100')
				elif row == 3:
					print('     CDS             complement(', i+1, '..', i+3, ')', sep='')
					print('                     /colour=100 100 100')
				'''
			outfile.write("//\n")
	
