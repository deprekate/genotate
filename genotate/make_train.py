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
import pathlib

class Translate:
	def __init__(self):
		nucs = ['t', 'c', 'a', 'g']
		codons = [a+b+c for a in nucs for b in nucs for c in nucs]
		amino_acids = 'FFLLSSSSYY#+CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'
		self.translate = dict(zip(codons, amino_acids))
		self.amino_acids = sorted(set(amino_acids))
		for c in '#+*':
			self.amino_acids.remove(c)

	def codon(self, codon):
		codon = codon.lower()
		if codon in self.translate:
			return self.translate[codon]
		else:
			return ''
	def counts(self, seq, strand):
		return Counter(self.seq(seq, strand))

	def frequencies(self, seq, strand):
		counts = self.counts(seq, strand)
		total = sum(counts.values())
		#for c in '#+*':
		#	del counts[c]
		#total = sum(counts.values())
		for aa in counts:
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

def nint(x):
	return int(x.replace('<','').replace('>',''))

def gc_content(seq):
	g = seq.count('g')
	c = seq.count('c')
	a = seq.count('a')
	t = seq.count('t')
	return (g+c) / (g+c+a+t)

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
				seq += line.decode("utf-8").rstrip().lower()
		contigs_dict[name] = seq.translate(base_trans)
	if '' in contigs_dict: del contigs_dict['']

	assert contigs_dict, "No DNA sequence found in the infile"
	return contigs_dict

def get_stops(infile):
	stops = dict()
	with open(infile) as fp:
		for line in fp:
			if line.startswith('     CDS '):
				pairs = [pair.split('..') for pair in re.findall(r"<*\d+\.\.>*\d+", line)]
				if line.rstrip().endswith(','):
					pairs.extend([pair.split('..') for pair in re.findall(r"<*\d+\.\.>*\d+", next(fp))])
				for pair in pairs:
					left,right = map(int, [ item.replace('<','').replace('>','') for item in pair ] )
					if pair[0] == '<1':
							left = right % 3 + 1
					if 'complement' in line:
						stops[left] = right
					else:
						stops[right - 2] = left
	return stops


def read_genbank(infile):
	dna = False
	coding_frame = dict()
	with open(infile) as fp:
		for line in fp:
			if line.startswith('     CDS '):
				direction = -1 if 'complement' in line else 1
				pairs = [pair.split('..') for pair in re.findall(r"<*\d+\.\.>*\d+", line)]
				# this is for features that continue on next line
				if line.rstrip().endswith(','):
					pairs.extend([pair.split('..') for pair in re.findall(r"<*\d+\.\.>*\d+", next(fp))])
				# this is for weird malformed features
				if ',1)' in line:
					pairs.append(['1','1'])

				######
				# this is limiting stuff to only non hypothetical genes
				if False:
					while not line.startswith('                     /product='):
						line = next(fp).lower()
					if 'hypot' in line or 'etical' in line or 'unchar' in line or ('orf' in line and 'orfb' not in line):
						continue
				######

				# loop over the feature recording its location
				remainder = 0
				for pair in pairs:
					left,right = map(nint, pair)
					if '<' in pair[0]:
						left = left + ((int(pairs[-1][-1])) - left -2) % 3
					for i in range(left-remainder,right-1,3):
						if i > 0:
							coding_frame[ +(i + 0) * direction ] = 1 #True
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
				dna += line[10:].rstrip().replace(' ','').lower()

	assert dna, "No DNA sequence found in the infile"

	for i, row in enumerate(get_windows(dna), start=1):
		pos = -((i+1)//2) if (i+1)%2 else ((i+1)//2)
		yield [coding_frame.get(pos, 2)] + [round(r, 5) for r in row]
		#yield [coding_frame.get(pos, 2)] + row
		
		#if pos in coding_frame:
		#	yield [coding_frame.get(pos, 2)] + [round(r, 5) for r in row]
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

def gcfp_freq(dna, strand):
	row = []
	for f in [0,1,2]:
		frame = dna[f::3]
		row.append( frame.count('G') + frame.count('C') )
	row = [count / sum(row) for count in row ] if sum(row) else [0,0,0]
	return row[::strand]

def nucl_fp(dna, strand):
	row = []
	for f in [0,1,2]:
		frame = dna[f::3]
		r = []
		r.append( frame.count('a') )
		r.append( frame.count('t') )
		r.append( frame.count('g') )
		r.append( frame.count('c') )
		r = [count / sum(r) for count in r ]
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
	

def single_window(dna, n, strand):
	'''
	get ONE window of 117 bases centered at the CODon
				.....COD.....   => translate => count aminoacids => [1,2,...,19,20]
	'''
	row = []
	translate = Translate()
	window = dna[ max( n%3 , n-57) : n+60]
	row.extend(nucl_freq(window, strand))
	#row.extend(nucl_fp(window, strand))
	#row.extend(gcfp_freq(window, strand))
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
	#row.extend(nucl_freq(window, strand))
	freqs = translate.frequencies(window, strand)
	for aa in translate.amino_acids:
		row.append(freqs.get(aa,0))
	# second
	window = dna[            n      : n+60 ]
	#row.extend(nucl_freq(window, strand))
	freqs = translate.frequencies(window, strand)
	for aa in translate.amino_acids:
		row.append(freqs.get(aa,0))
	return row

def glob_window(dna, n, strand):
	'''
	lol go crazy with windows
	'''
	row = []
	window = dna[ max( n%3 , n-57) : n+60]
	#row = [gc_content(window), n+1] + nucl_freq(window, strand) + gcpos_freq(window, strand)
	row.extend(nucl_freq(window, strand))
	for j in [0,1,2]:
		row.extend(single_window(dna, n+(j*strand),  strand )[4:])
		row.extend(single_window(dna, n+(j*strand), -strand )[4:])
	return row	

def dglob_window(dna, n, strand):
	row = []
	window = dna[ max( n%3 , n-57 ) : n+3  ]
	row.extend(nucl_freq(window, strand))
	window = dna[            n      : n+60 ]
	row.extend(nucl_freq(window, strand))
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
			yield [gc] + single_window(dna, n+f, +1)
			yield [gc] + single_window(dna, n+f, -1 )
			#yield [gc] + double_window(dna, n+f, +1)
			#yield [gc] + double_window(dna, n+f, -1 )
			#yield [gc] + glob_window(dna, n+f, +1 )
			#yield [gc] + glob_window(dna, n+f, -1 )
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
	parser.add_argument('-t', '--type', action="store", default="single", dest='outfmt', help='type of window [single]', choices=['single','double','glob'])
	args = parser.parse_args()

	# print the column header and quit
	if args.labels:
		translate = Translate()
		sys.stdout.write('\t'.join(['TYPE','ID', 'GC'] + [aa for aa in translate.amino_acids]))
		sys.stdout.write('\n')

	#contigs = read_fasta(args.infile)
	#for row in get_windows(contigs[list(contigs.keys())[0]]):
	#	args.outfile.write('\t'.join(map(str,row)))
	#	args.outfile.write('\n')
	#exit()

	if os.path.isdir(args.infile):
		#raise NotImplementedError('Running on multiple files in a directory')
		for infile in os.listdir(args.infile):
			for row in read_genbank(os.path.join(args.infile, infile)):
				args.outfile.write('\t'.join(map(str,row)))
				args.outfile.write('\n')
	else:
		if pathlib.Path(args.infile).suffix in ['gb', 'gbk']:
			infile = read_genbank(args.infile)
		else:
			infile = get_windows(list(read_fasta(args.infile).values())[0])
		for row in infile:
			args.outfile.write('\t'.join(map(str,row)))
			args.outfile.write('\n')





