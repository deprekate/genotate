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
from genotate.features import Features

#from genotate.windows import get_windows
from genotate.make_train import get_windows
#from genotate.mt import get_windows

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

def smo(data):
	out = np.zeros_like(data)
	for i in range(6):
		for j in range(3):
			out[i::6,j] = smooth_line(data[i::6,j])
	return out


def smooth_line(x,window_len=101,window='hamming'):
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

def smoo(data):
	out = np.zeros_like(data)
	for i in range(len(data)):
		window = data[ max(i-19, i % 6) : i+20 : 6 ]
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
	args = parser.parse_args()
	'''
	if args.labels: print("\t".join(['ID','TYPE','GC'] + translate.amino_acids))
		exit()
	'''

	outfile = args.outfile

	ckpt_reader = tf.train.load_checkpoint(args.model)
	n = len(ckpt_reader.get_tensor('layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE'))
	model = mm.create_model_d(n)
	model.load_weights(args.model).expect_partial()

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
											dtype=tf.float32
											)
										)
								).batch(32)
		#for feature in dataset.take(1):
		#	print( feature )
		#exit()
		p = model.predict(dataset)
		#p = smo(p)
		#p = smoo(p)
		#p = best(p)
		'''
		for i in range(0,len(p), 6):
			for j in range(6):
				print(p[i+j], end='\t')
			print()
		exit()
		'''
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
		if args.genes:

			nc = np.array([
                p[0::6,0],
                p[1::6,0],
				p[2::6,0],
                p[3::6,0],
                p[4::6,0],
                p[5::6,0],
                p[0::6,2],
                p[1::6,2],
				p[2::6,2],
                p[3::6,2],
                p[4::6,2],
                p[5::6,2]
				])

			signal = np.array([
                p[0::6,1],
                p[1::6,1],
				p[2::6,1],
                p[3::6,1],
                p[4::6,1],
                p[5::6,1],
				np.mean(nc, axis=0)
                ]).T
			import ruptures as rpt
			# detection
			#algo = rpt.Pelt(model="rbf").fit(signal)
			algo = rpt.KernelCPD(kernel="linear", min_size=30).fit(signal[:-3,:])
			result = algo.predict(pen=15)
			
			frame = {0:1, 1:-1, 2:2, 3:-2, 4:3, 5:-3, 6:0}
			last = 0
			for loc in [item * 3 for item in result]:
				#print(last,loc, frame[ np.argmax(signal[last//3 : loc//3,].mean(axis=0)) ], sep='\t') ;  last = loc ; continue
				f = frame[ np.argmax(signal[last//3 : loc//3,].mean(axis=0)) ]
				left = (last//3)*3 + abs(f)
				right = (loc//3)*3 + abs(f) + 2
				if f > 0:
					outfile.write("     CDS             " + str(left) + ".." + str(right))
					outfile.write('\n')
					outfile.write("                     /nearest_start=" + str(locations.nearest_start( left - 1 )) )
					outfile.write('\n')
					outfile.write("                     /nearest_stopp=" + str(locations.nearest_stop( right - 3 )) )
					outfile.write('\n')
				elif f < 0:
					outfile.write("     CDS             complement(" + str(left) + ".." + str(right) + ")")
					outfile.write('\n')
					outfile.write("                     /nearest_start=" + str(locations.nearest_start( right - 3 , forward=False)) )
					outfile.write('\n')
					outfile.write("                     /nearest_stopp=" + str(locations.nearest_stop( left - 1 , forward=False)) )
					outfile.write('\n')
				last = loc
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
	
