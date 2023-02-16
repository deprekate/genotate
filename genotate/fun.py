import os
import sys
from itertools import zip_longest
import gc

import numpy as np
import tensorflow as tf
from genbank.file import File

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def parse_genbank(infile):
	gc.collect()
	genbank = File(infile.decode())
	#A = np.zeros([len(genbank.dna()),99])
	# intergenic
	for locus in genbank:
		dna = locus.seq()
		Y = np.zeros([len(dna)*2+7,3],dtype=np.uint8)
		Y[:,2] = 1
		# label the positions
		positions = dict()
		for feature in locus.features(include=['CDS']):
			s = (feature.strand * -1 + 1) >> 1
			locations = feature.codon_locations()
			if feature.partial() == 'left': next(locations)
			for i in locations:
				# coding
				Y[2*i[0]+0+s,1] = 1
				# noncoding
				Y[2*i[0]+0+s,0] = 1
				Y[2*i[0]+1+s,0] = 1
				Y[2*i[0]+2+s,0] = 1
				Y[2*i[0]+3+s,0] = 1
				Y[2*i[0]+4+s,0] = 1
				Y[2*i[0]+5+s,0] = 1
				Y[2*i[0]+0+s,2] = 0
				Y[2*i[0]+1+s,2] = 0
				Y[2*i[0]+2+s,2] = 0
				Y[2*i[0]+3+s,2] = 0
				Y[2*i[0]+4+s,2] = 0
				Y[2*i[0]+5+s,2] = 0
		Y[Y[:,1]==1,0] = 0
		forward = np.zeros(48+len(dna)+50,dtype=np.uint8)
		reverse = np.zeros(48+len(dna)+50,dtype=np.uint8)
		for i,base in enumerate(dna):
			#if base in 'acgt':
			forward[i+48] = ((ord(base) >> 1) & 3) + 1
			reverse[i+48] = ((forward[i+48] - 3) % 4) + 1
		#a = np.zeros([6, 100], dtype=int)
		# leave this here for numpy < 1.20 backwards compat
		#forfor = np.concatenate(( forward, forward[:-1] ))
		#L = len(forward)
		#n = forfor.strides[0]
		#f = np.lib.stride_tricks.as_strided(forfor[L-1:], (L,L), (-n,n))
		'''
		# this is creates one BIG numpy array
		X = np.zeros([len(dna)*2  ,99],dtype=np.uint8)
		X[0::2,] = np.lib.stride_tricks.sliding_window_view(forward,99)
		X[1::2,] = np.lib.stride_tricks.sliding_window_view(reverse,99)[:,::-1]
		#A[I:i+1,:] = w
		#I = i
		yield X,Y[:-7]
		'''
		# this splits the BIG numpy array into two to limit ram usage
		X = np.zeros([len(dna) + len(dna)%2  ,99],dtype=np.uint8)
		middle = len(dna)//2  + (len(dna) % 2 > 0)
		X[0::2,] = np.lib.stride_tricks.sliding_window_view(forward[:middle+98],99)
		X[1::2,] = np.lib.stride_tricks.sliding_window_view(reverse[:middle+98],99)[:,::-1]
		yield X,Y[:len(X),:]
		X[0::2,] = np.lib.stride_tricks.sliding_window_view(forward[middle-(len(dna)%2):],99)
		X[1::2,] = np.lib.stride_tricks.sliding_window_view(reverse[middle-(len(dna)%2):],99)[:,::-1]
		yield X[2*(len(dna)%2):,:],Y[len(X):-7,:]
	del genbank

class GenDataset(tf.data.Dataset):
	#@tf.function
	def __new__(self, infile):
		spec = (tf.TensorSpec(shape = (None,99), dtype = tf.experimental.numpy.int8),
				tf.TensorSpec(shape = (None, 3), dtype = tf.experimental.numpy.int8))
		#return tf.py_function(self.parse_genbank, infile, Tout=spec )
		return tf.data.Dataset.from_generator(
			parse_genbank,
			#output_signature = tf.TensorSpec(shape = (None,99), dtype = tf.int32),
			output_signature = spec,
			args=(infile,)
		) #.unbatch() #.flat_map(tf.data.Dataset.from_tensor_slices),




class GenomeDataset(tf.data.Dataset):
	#@tf.function
	def __new__(self, infile):
		return tf.data.Dataset.from_generator(
			self.iter_genbank,
			output_signature = tf.TensorSpec(shape = (6,100), dtype = tf.int32),
			#output_signature = (tf.TensorSpec(shape = ( ), dtype = tf.int32),tf.TensorSpec(shape = (99, ), dtype = tf.int32)),
			args=(infile,)
		)
	def strided_method(self,ar):
		a = np.concatenate(( ar, ar[:-1] ))
		L = len(ar)
		n = a.strides[0]
		return np.lib.stride_tricks.as_strided(a[L-1:], (L,L), (-n,n))

	def iter_genbank(infile):
		for locus in File(infile.decode()):
			# label the positions
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
			#for k,v in positions.items():
			#	print(k,v)
			#exit()
			dna = locus.seq()
			#at_skew = np.array(skew(dna, 'at'))
			#gc_skew = np.array(skew(dna, 'gc'))
			forward = np.zeros(48+len(dna)+50)
			reverse = np.zeros(48+len(dna)+50)
			for i,base in enumerate(dna):
				#if base in 'acgt':
				forward[i+48] = ((ord(base) >> 1) & 3) + 1
				reverse[i+48] = ((forward[i+48] - 3) % 4) + 1
			a = np.zeros([6, 100], dtype=int)
			# leave this here for numpy < 1.20 backwards compat
			#forfor = np.concatenate(( forward, forward[:-1] ))
			#L = len(forward)
			#n = forfor.strides[0]
			#f = np.lib.stride_tricks.as_strided(forfor[L-1:], (L,L), (-n,n))
			#w = np.lib.stride_tricks.sliding_window_view(a,99)
			#a[:,100] = locus.gc_content() 
			for n in range(0, len(dna)-2, 3):
				#pos = n//100
				i = n + 0
				#yield  positions.get(  i, 2) , forward[i : i+99 ]       
				#yield  positions.get( -i, 2) , forward[i : i+99 ][::-1] 
				a[0,0] = positions.get( i, 2)
				a[1,0] = positions.get(-i, 2)
				a[0,1:100] = forward[i : i+99 ]
				a[1,1:100] = reverse[i : i+99 ][::-1]
				i = n + 1
				#yield  positions.get(  i, 2) , forward[i : i+99 ]       
				#yield  positions.get( -i, 2) , forward[i : i+99 ][::-1] 
				a[2,0] = positions.get( i, 2)
				a[3,0] = positions.get(-i, 2)
				a[2,1:100] = forward[i : i+99 ]
				a[3,1:100] = reverse[i : i+99 ][::-1]
				i = n + 2
				#yield  positions.get(  i, 2) , forward[i : i+99 ]      
				#yield  positions.get( -i, 2) , forward[i : i+99 ][::-1] 
				a[4,0] = positions.get( i, 2)
				a[5,0] = positions.get(-i, 2)
				a[4,1:100] = forward[i : i+99 ]
				a[5,1:100] = reverse[i : i+99 ][::-1]
				yield a
