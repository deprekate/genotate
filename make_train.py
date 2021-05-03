#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import io
import sys
import re
from math import log
import random
import argparse
from argparse import RawTextHelpFormatter
from collections import Counter

class Translate:
	def __init__(self):
		nucs = ['T', 'C', 'A', 'G']
		codons = [a+b+c for a in nucs for b in nucs for c in nucs]
		amino_acids = 'FFLLSSSSYY#+CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'
		self.translate = dict(zip(codons, amino_acids))
		self.amino_acids = sorted(set(amino_acids))
		for c in '#+*':
			self.amino_acids.remove(c)

	def codon(self, codon):
		codon = codon.upper()
		if codon in self.translate:
			return self.translate[codon]
		else:
			return ''
	def counts(self, seq, strand):
		return Counter(self.seq(seq, strand))

	def frequencies(self, seq, strand):
		counts = self.counts(seq, strand)
		#total = sum(counts.values())
		for c in '#+*':
			del counts[c]
		total = sum(counts.values())
		for aa in counts:
			#counts[aa] = round(counts[aa] / total, 4)
			counts[aa] = counts[aa] / total
		return counts

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
		seq_dict = {'A':'T','T':'A','G':'C','C':'G',
					'N':'N',
					'R':'Y','Y':'R','S':'S','W':'W','K':'M','M':'K',
					'B':'V','V':'B','D':'H','H':'D'}
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
	g = seq.count('G')
	c = seq.count('C')
	a = seq.count('A')
	t = seq.count('T')
	return (g+c) / (g+c+a+t)

def our_generator():
    for i in range(1000):
      x = np.random.rand(28,28)
      y = np.random.randint(1,10, size=1)
      yield x,y

def read_fasta(filepath, base_trans=str.maketrans('','')):
	contigs_dict = dict()
	name = seq = ''

	lib = gzip if filepath.endswith(".gz") else io
	with lib.open(filepath, mode="rb") as f:
		for line in f:
			if line.startswith(b'>'):
				contigs_dict[name] = seq
				name = line[1:].decode("utf-8").split()[0]
				seq = ''
			else:
				seq += line.decode("utf-8").rstrip().upper()
		contigs_dict[name] = seq.translate(base_trans)
	if '' in contigs_dict: del contigs_dict['']
	return contigs_dict

def read_genbank(infile):
	dna = False
	coding_frame = dict()
	with open(infile) as fp:
		for line in fp:
			if line.startswith('     CDS '):
				direction = -1 if 'complement' in line else 1
				pairs = [pair.split('..') for pair in re.findall(r"<*\d+\.\.>*\d+", line)]
				if ',1)' in line:
					pairs.append(['1','1'])
				remainder = 0
				for pair in pairs:
					left,right = map(int, [ item.replace('<','').replace('>','') for item in pair ] )
					if pair[0] == '<1':
						left = right % 3 + 1
					for i in range(left-remainder,right-1,3):
						coding_frame[ +(i + 0) * direction ] = 1     #True
						if +(i + 1) * direction not in coding_frame:
							coding_frame[ +(i + 1) * direction ] = 0 #False
						if +(i + 2) * direction not in coding_frame:
							coding_frame[ +(i + 2) * direction ] = 0 #False
						if -(i + 0) * direction not in coding_frame:
							coding_frame[ -(i + 0) * direction ] = 0 #False
						if -(i + 1) * direction not in coding_frame:
							coding_frame[ -(i + 1) * direction ] = 0 #False
						if -(i + 2) * direction not in coding_frame:
							coding_frame[ -(i + 2) * direction ] = 0 #False
						remainder = right-2 - i
				if remainder and ">" not in pair[1]:
					raise ValueError("Out of frame: ( %s , %s )" % tuple(pair))
			elif line.startswith('ORIGIN'):
				dna = ''
			elif dna != False:
				dna += line[10:].rstrip().replace(' ','').upper()

	assert dna, "No DNA sequence found in the infile"

	for i, row in enumerate(get_windows(dna), start=1):
		pos = -((i+1)//2) if (i+1)%2 else ((i+1)//2)
		yield [coding_frame.get(pos, 0)] + [round(r, 3) for r in row]
		'''
		if coding_frame.get(i, 0) and coding_frame.get(-i, 0):
			#yield [int(random.random() * 2) * 2 - 1] + row
			#yield [int(random.random() * 2) + 1] + row
			yield [3] + row
		elif coding_frame.get(i, 0):
			yield [1] + row
		elif coding_frame.get(-i, 0):
			yield [2] + row
		else:
			yield [0] + row
		'''

def gcpos_freq(dna, strand):
	row = []
	for f in [0,1,2]:
		frame = dna[f::3]
		row.append( frame.count('G') + frame.count('C') )
	row = [count / sum(row) for count in row ]
	return row[::strand]

def nucl_freq(dna, strand):
	n = len(dna) 
	a = dna.count('A') / n
	t = dna.count('T') / n
	g = dna.count('G') / n
	c = dna.count('C') / n
	if strand > 0:
		return [a, t, g, c]
	else:
		return [t, a, c, g]
	

def single_window(dna, n, strand):
	'''
	get ONE window of 117 bases centered at the CODon
				.....COD.....   => translate => count aminoacids => [1,2,...,19,20]
	'''
	row = []
	translate = Translate()
	window = dna[ max( n%3 , n-57) : n+60]
	freqs = translate.frequencies(window, strand)
	for aa in translate.amino_acids:
		row.append(freqs.get(aa,0))
	return row

def double_window(dna, n, strand):
	'''
	get TWO windows of 60 bases centered at the CODon
				.....COD        => translate => count aminoacids => [1,2,...,19,20]
					 COD.....   => translate => count aminoacids => [1,2,...,19,20]
																		  |
																		  V
															[1,2,...,19,20 , 1,2,...,19,20]
	'''
	row = []
	translate = Translate()
	# first
	window = dna[ max( n%3 , n-57 ) : n+3  ]
	freqs = translate.frequencies(window, strand)
	for aa in translate.amino_acids:
		row.append(freqs.get(aa,0))
	# second
	window = dna[            n      : n+60 ]
	freqs = translate.frequencies(window, strand)
	for aa in translate.amino_acids:
		row.append(freqs.get(aa,0))
	return row

def glob_window(dna, n, strand):
	'''
	lol go crazy with windows
	'''
	#row = []
	window = dna[ max( n%3 , n-57) : n+60]
	#row = [gc_content(window), n+1] + nucl_freq(window, strand) + gcpos_freq(window, strand)
	row = nucl_freq(window, strand)
	#row = [gc_content(window)] + [ count / sum(counts) for count in counts ]
	#row = [ count / sum(counts) for count in counts ]
	for j in [0,1,2]:
		row.extend(single_window(dna, n+(j*strand),  strand ))
		row.extend(single_window(dna, n+(j*strand), -strand ))
	return row	

def dglob_window(dna, n, strand):
	row = []
	for j in [0,1,2]:
		row.extend(double_window(dna, n+(j*strand),  strand ))
		row.extend(double_window(dna, n+(j*strand), -strand ))
	return row	

def get_windows(dna):
	'''
	This method takes as input a string of the nucleotides, and returns
	the amino-acid frequencies of a window centered at each potential codon
	position.  The positions are forward and reverse interlaced and so the
	rows follow the pattern:
		+1 -1 +2 -2 +3 -3 +4 -4 +5 -5 +6 -6 +7 -7 etc

	'''
	# this is to fix python variable passing issues
	if type(dna) is not str:
		dna = dna.decode()

	#translate = Translate()
	gc = gc_content(dna) 

	# get the aminoacid frequency window
	#args = lambda: None
	for n in range(0, len(dna)-2, 3):
		for f in [0,1,2]:
			#yield [gc] + single_window(dna, n+f, False)
			#yield [gc] + single_window(dna, n+f, True )
			#yield [gc] + double_window(dna, n+f, False)
			#yield [gc] + double_window(dna, n+f, True )
			yield [gc] + glob_window(dna, n+f, +1 )
			yield [gc] + glob_window(dna, n+f, -1 )
			#yield [gc] + dglob_window(dna, n+f, +1 )
			#yield [gc] + dglob_window(dna, n+f, -1 )



if __name__ == '__main__':
	usage = 'make_train.py [-opt1, [-opt2, ...]] infile'
	parser = argparse.ArgumentParser(description='', formatter_class=RawTextHelpFormatter, usage=usage)
	parser.add_argument('infile', type=is_valid_file, help='input file in genbank format')
	parser.add_argument('-o', '--outfile', action="store", default=sys.stdout, type=argparse.FileType('w'), help='where to write output [stdout]')
	#parser.add_argument('-w', '--window', action="store", type=int, default=120,  help='The size of the window')
	parser.add_argument('-l', '--labels', action="store_true", help=argparse.SUPPRESS)
	parser.add_argument('--ids', action="store", help=argparse.SUPPRESS)
	args = parser.parse_args()

	# print the column header and quit
	if args.labels:
		translate = Translate()
		sys.stdout.write('\t'.join(['TYPE','ID', 'GC'] + [aa for aa in translate.amino_acids]))
		sys.stdout.write('\n')

	if os.path.isdir(args.infile):
		#raise NotImplementedError('Running on multiple files in a directory')
		for infile in os.listdir(args.infile):
			for row in read_genbank(os.path.join(args.infile, infile)):
				args.outfile.write('\t'.join(map(str,row)))
				args.outfile.write('\n')
	else:
		for row in read_genbank(args.infile):
			args.outfile.write('\t'.join(map(str,row)))
			args.outfile.write('\n')





