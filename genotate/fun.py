import os
import sys

import numpy as np
import tensorflow as tf
from genbank.file import File

class GenomeDataset(tf.data.Dataset):
	@tf.function
	def __new__(cls, infile):
		return tf.data.Dataset.from_generator(
			cls.iter_genbank,
			output_signature = tf.TensorSpec(shape = (6,100), dtype = tf.int32),
			#output_signature = (tf.TensorSpec(shape = (6,99), dtype = tf.int32),tf.TensorSpec(shape = (6,3), dtype = tf.int32)),
			args=(infile,)
		)

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
			#a[:,100] = locus.gc_content() 
			for n in range(0, len(dna)-2, 3):
				#pos = n//100
				i = n + 0
				a[0,0] = positions.get( i, 2)
				a[1,0] = positions.get(-i, 2)
				a[0,1:100] = forward[i : i+99 ]
				a[1,1:100] = reverse[i : i+99 ][::-1]
				i = n + 1
				a[2,0] = positions.get( i, 2)
				a[3,0] = positions.get(-i, 2)
				a[2,1:100] = forward[i : i+99 ]
				a[3,1:100] = reverse[i : i+99 ][::-1]
				i = n + 2
				a[4,0] = positions.get( i, 2)
				a[5,0] = positions.get(-i, 2)
				a[4,1:100] = forward[i : i+99 ]
				a[5,1:100] = reverse[i : i+99 ][::-1]
				#a[2*f  ,101] =  at_skew[pos]
				#a[2*f+1,101] = -at_skew[pos]
				#a[2*f  ,102] =  gc_skew[pos]
				#a[2*f+1,102] = -gc_skew[pos]
				yield a
