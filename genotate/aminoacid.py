#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
import re
from math import log
import gzip
import argparse
from argparse import RawTextHelpFormatter
from collections import Counter
from itertools import zip_longest # for Python 3.x

def grouper(iterable, n,  padvalue=None):
    "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
    return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)


import numpy as np
import matplotlib.pyplot as plt

#sign = lambda x: (1, -1)[x<0]

#import faulthandler
#sys.settrace

from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE,SIG_DFL)

class Translate:
	def __init__(self):
		nucs = ['t', 'c', 'a', 'g']
		self.codons = [a+b+c for a in nucs for b in nucs for c in nucs]
		amino_acids = 'FFLLSSSSYY#+CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'
		self.translate = dict(zip(self.codons, amino_acids))
		self.amino_acids = sorted(set(amino_acids))
		#for c in '#+*':
		#	self.amino_acids.remove(c)

	def codon(self, codon):
		codon = codon.lower()
		if codon in self.translate:
			return self.translate[codon]
		else:
			return ''
	def counts(self, seq, strand):
		aa = self.seq(seq, strand)
		return Counter(aa)

	def frequencies(self, seq, strand):
		counts = self.counts(seq, strand)
		total = sum(counts.values())
		#for c in '#+*':
		#	del counts[c]
		#total = sum(counts.values())
		for aa in counts:
			counts[aa] = counts[aa] / total
		return counts

	def codon_entropy(self, seq, strand):
		row = []
		codons = dict()
		for codon in re.findall('...', seq): 
			if strand > 0:
				codons[codon] = codons.get(codon, 0) + 1
			else:
				codons[self.rev_comp(codon)] = codons.get(self.rev_comp(codon), 0) + 1
		counts = dict()
		for codon in codons:
			counts.setdefault(self.translate[codon], []).append(codons[codon])
		for aa in self.amino_acids:
			h = 0.0
			count = counts.get(aa, [])
			for p in count:
				h += (p/sum(count)) * log(p/sum(count))
			row.append(-h)
		print(row)
		exit()

	def structure(self, seq, strand):
		prot = self.seq(seq, strand)
		encode = {
				**{aa : ''	 for aa in '#*+'},
				**{aa : 'u'  for aa in 'ACDEFGHIKLMNPQRSTVWY'},
				**{aa : 'a'  for aa in 'AELMKCH'},
				**{aa : 'na' for aa in 'YSGP'},
				**{aa : 'b'  for aa in 'TVIFWY'}
				 } 
		counts = {
				'' : 0,
				**{one : 0 for one in 'uab'},
				**{one+two : 0 for one in 'uab' for two in 'uab'}
				}
		tot = 0
		for i in range(len(prot)-1):
			counts[ (encode[prot[i]] + encode[prot[i+1]]) ] += 1
			tot += 1
		row = []
		for key in [one+two for one in 'uab' for two in 'uab']:
			row.append(counts[key] / tot)	
		print(prot)
		print(row)
		return row
	
	def image(self, seq,strand):
		encode = { letter:i for i,letter in enumerate('CTSAGPEQKRDNHYFMLVIW*+#') }
		A = np.zeros([23,23])
		prot = self.seq(seq, strand)
		for a,b in zip(prot, prot[1:]):
			A[encode[a]][encode[b]] += 1
		A = A / np.sum(A)

		plt.imshow(A, cmap="gray",vmin=0, vmax=0.5)
		plt.savefig('img/' + prot + '.png')



	def array(self, seq, strand):
		#encode = { letter:i for i,letter in enumerate('#WYFVIJLMCZEQKRHBDNXATSGP') }
		encode = { letter:i for i,letter in enumerate('#CTSAGPEQKRDNHYFMLVIW') }
		encode['*'] = 0
		encode['+'] = 0
		
		prot = self.seq(seq, strand)
		array = [[0] * 40] * 21
		for i,aa in enumerate(prot):
			array[encode[aa]][i] = 1
		return array

	def dipeps(self, seq, strand):
		prot = self.seq(seq, strand)
		counts = { (a,b):0 for a in self.amino_acids for b in self.amino_acids }
		for i in range(len(prot)-2):
			counts[ ( prot[i] , prot[i+2] ) ] += 1
		t = sum(counts.values()) if sum(counts.values()) else 1
		return [ counts[(a,b)]/t for a in self.amino_acids for b in self.amino_acids ]

	def trimers(self, seq, strand):
		prot = self.seq(seq, strand)
		counts = { (a,b,c):0 for a in self.amino_acids for b in self.amino_acids for c in self.amino_acids }
		for i in range(len(prot)-2):
			counts[ ( prot[i] , prot[i+1], prot[i+2] ) ] += 1
		t = sum(counts.values()) if sum(counts.values()) else 1
		return [ counts[(a,b,c)]/t for a in self.amino_acids for b in self.amino_acids for c in self.amino_acids ]

	def dimers(self, seq, strand):
		prot = self.seq(seq, strand)
		counts = { (a,b):0 for a in self.amino_acids for b in self.amino_acids }
		for i in range(len(prot)-1):
			counts[ ( prot[i] , prot[i+1] ) ] += 1
		t = sum(counts.values()) if sum(counts.values()) else 1
		return [ counts[(a,b)]/t for a in self.amino_acids for b in self.amino_acids ]

	def tricodings(self, seq, strand):
		prot = self.seq(seq, strand)
		encode = {
				 '#':0, '+':0, '*':0,
				 'A': 1, 'I': 1, 'L': 1, 'M': 1, 'V': 1,
				 'N': 2, 'Q': 2, 'S': 2, 'T': 2,
				 'G': 3, 'P': 3,
				 'C': 4,
				 'H': 5, 'K': 5, 'R': 5,
				 'D': 6, 'E': 6,
				 'F': 7, 'W': 7, 'Y': 7
				 } 
		counts = { (a,b,c):0 for a in range(8) for b in range(8) for c in range(8) }
		for i in range(len(prot)-2):
			counts[ (encode[prot[i]] , encode[prot[i+1]], encode[prot[i+2]]) ] += 1
		'''
		for key in list(counts.keys()):
			if 0 in key:
				del counts[key]
		'''
		t = sum(counts.values()) if sum(counts.values()) else 1
		return [ counts[(a,b,c)]/t for a in range(8) for b in range(8) for c in range(8) ]


	def codings(self, seq, strand):
		prot = self.seq(seq, strand)
		encode = {
				 '#':0, '+':0, '*':0,
				 'A': 1, 'I': 1, 'L': 1, 'M': 1, 'V': 1,
				 'N': 2, 'Q': 2, 'S': 2, 'T': 2,
				 'G': 3, 'P': 3,
				 'C': 4,
				 'H': 5, 'K': 5, 'R': 5,
				 'D': 6, 'E': 6,
				 'F': 7, 'W': 7, 'Y': 7
				 } 
		counts = { a:0 for a in range(8) }
		for i in range(len(prot)):
			counts[ encode[prot[i]] ] += 1
		t = sum(counts.values()) if sum(counts.values()) else 1
		return [ counts[a]/t for a in range(8) ]
	
	def dicodings(self, seq, strand):
		prot = self.seq(seq, strand)
		encode = {
				 '#':0, '+':0, '*':0,
				 'A': 1, 'I': 1, 'L': 1, 'M': 1, 'V': 1,
				 'N': 2, 'Q': 2, 'S': 2, 'T': 2,
				 'G': 3, 'P': 3,
				 'C': 4,
				 'H': 5, 'K': 5, 'R': 5,
				 'D': 6, 'E': 6,
				 'F': 7, 'W': 7, 'Y': 7
				 } 
		counts = { (one,two):0 for one in range(8) for two in range(8) }
		for i in range(len(prot)-1):
			counts[ (encode[prot[i]] , encode[prot[i+1]]) ] += 1
		'''
		for key in list(counts.keys()):
			if 0 in key:
				del counts[key]
		'''
		t = sum(counts.values()) if sum(counts.values()) else 1
		return [ counts[(a,b)]/t for a in range(8) for b in range(8) ]

	def seq(self, seq, strand):
		aa = ''
		if strand > 0:
			for i in range(0, len(seq), 3):
				aa += self.codon(seq[i:i+3])
			return aa
		else:
			for i in range(0, len(seq), 3):
				aa += self.codon(self.rev_comp(seq[i:i+3]))
			return aa[::-1]

	def rev_comp(self, seq):
		seq_dict = {'a':'t','t':'a','g':'c','c':'g',
					'n':'n',
					'r':'y','y':'r','s':'s','w':'w','k':'m','m':'k',
					'b':'v','v':'b','d':'h','h':'d'}
		return "".join([seq_dict[base] for base in reversed(seq)])

	def edp(self, seq, strand):
		"""Calculate entropy"""
		H = 0
		counts = self.counts(seq, strand)
		for aa in self.amino_acids:
			p = -counts[aa]*log(counts[aa]) if counts[aa] else 0
			counts[aa] = p
			H += p
		for aa in self.amino_acids:
			counts[aa] /= H
		return counts


def is_valid_file(x):
	if not os.path.exists(x):
		raise argparse.ArgumentTypeError("{0} does not exist".format(x))
	return x

def same_frame(a,b):
	return (a)%3 == (b-2)%3


def gc_content(seq):
	a = seq.count('a')
	c = seq.count('c')
	g = seq.count('g')
	t = seq.count('t')
	tot = a+c+g+t if a+c+g+t else 1
	return (c+g) / tot


def codon_usage(dna, strand):
	row = []
	codons = dict()
	for f in [0,1,2]:
		frame = dna[f::3]
		for codon in re.findall('...',dna): 
			if strand > 0:
				codons[codon] = codons.get(codon, 0) + 1
			else:
				codons[rev_comp(codon)] = codons.get(rev_comp(codon), 0) + 1
	return codons


def gc_fp(dna, strand):
	row = []
	for f in [0,1,2]:
		frame = dna[f::3]
		row.append( frame.count('g') + frame.count('c') )
	row = [count / sum(row) for count in row ] if sum(row) else [0,0,0]
	return row[::strand]

def nucl_fp(dna, strand):
	row = []
	for f in [0,1,2]:
		frame = dna[f::3]
		r = []
		r.append( frame.count('a') )
		r.append( frame.count('c') )
		r.append( frame.count('g') )
		r.append( frame.count('t') )
		t = sum(r) if sum(r) else 1
		r = [count / t for count in r ]
		row.extend(r)
	return row[::strand]

def nucl_freq(dna, strand):
	n = len(dna) 
	a = dna.count('a') / n if n else 0
	c = dna.count('c') / n if n else 0
	g = dna.count('g') / n if n else 0
	t = dna.count('t') / n if n else 0
	if strand > 0:
		return [a, c, g, t]
	else:
		return [t, g, c, a]

						
	

if __name__ == '__main__':
	usage = '%s [-opt1, [-opt2, ...]] infile' % __file__
	parser = argparse.ArgumentParser(description='', formatter_class=RawTextHelpFormatter, usage=usage)
	parser.add_argument('infile', type=is_valid_file, help='input file in genbank format')
	parser.add_argument('-o', '--outfile', action="store", default=sys.stdout, type=argparse.FileType('w'), help='where to write output [stdout]')
	#parser.add_argument('-w', '--window', action="store", type=int, default=120,  help='The size of the window')
	parser.add_argument('-l', '--labels', action="store_true", help=argparse.SUPPRESS)
	parser.add_argument('--ids', action="store", help=argparse.SUPPRESS)
	parser.add_argument('-t', '--type', action="store", default="single", dest='outfmt', help='type of window [single]', choices=['single','double','glob'])
	args = parser.parse_args()






