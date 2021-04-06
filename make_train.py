#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import io
import sys
import re
from math import log
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
	def counts(self, seq, rev=False):
		return Counter(self.seq(seq, rev=rev))

	def frequencies(self, seq, rev=False):
		counts = self.counts(seq, rev=rev)
		for c in '#+*':
			del counts[c]
		total = sum(counts.values())
		for aa in counts:
			counts[aa] = round(counts[aa] / total , 4)
			#counts[aa] = round(counts[aa] / 16 , 4)
		return counts

	def seq(self, seq, rev=False):
		aa = ''
		if rev:
			for i in range(0, len(seq), 3):
				aa += self.codon(self.rev_comp(seq[i:i+3]))
			return aa[::-1]
		else:
			for i in range(0, len(seq), 3):
				aa += self.codon(seq[i:i+3])
			return aa

	def rev_comp(self, seq):
		seq_dict = {'A':'T','T':'A','G':'C','C':'G',
					'N':'N',
					'R':'Y','Y':'R','S':'S','W':'W','K':'M','M':'K',
					'B':'V','V':'B','D':'H','H':'D'}
		return "".join([seq_dict[base] for base in reversed(seq)])

	def edp(self, seq, rev=False):
		"""Calculate entropy"""
		H = 0
		counts = self.counts(seq, rev=rev)
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
	return round( (g+c) / (g+c+a+t) , 3)

def our_generator():
    for i in range(1000):
      x = np.random.rand(28,28)
      y = np.random.randint(1,10, size=1)
      yield x,y

def read_fasta(filepath, base_trans=str.maketrans('','')):
	contigs_dict = dict()
	name = ''
	seq = ''

	lib = gzip if filepath.endswith(".gz") else io
	with lib.open(filepath, mode="rb") as f:
		for line in f:
			if line.startswith(b'>'):
				contigs_dict[name] = seq
				name = line[1:].decode("utf-8").split()[0]
				seq = ''
			else:
				#seq += line.replace("\n", "").upper()
				seq += line[:-1].decode("utf-8").upper()
		contigs_dict[name] = seq.translate(base_trans)
	if '' in contigs_dict: del contigs_dict['']
	return contigs_dict

def read_genbank(infile):
	dna = ''
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
						coding_frame[ +(i + 0) * direction ] = 2 #True
						coding_frame[ +(i + 1) * direction ] = 1 #False
						coding_frame[ +(i + 2) * direction ] = 1 #False
						coding_frame[ -(i + 0) * direction ] = 1 #False
						coding_frame[ -(i + 1) * direction ] = 1 #False
						coding_frame[ -(i + 2) * direction ] = 1 #False
						remainder = right-2 - i
				if remainder and ">" not in pair[1]:
					raise ValueError("Out of frame: ( %s , %s )" % tuple(pair))
			elif line.startswith('ORIGIN'):
				dna = '\n'
			elif dna:
				line = line[10:].replace(' ','')
				dna += line.upper()

	dna = dna.replace('\n', '')

	gc = gc_content(dna) 

	for i, row in enumerate(get_windows(dna), start=1):
		pos = -((i+1)//2) if (i+1)%2 else ((i+1)//2)
		yield [coding_frame.get(pos, 0)] + row

def get_windows(dna):
	'''
	This method takes as input a string of the nucleotides, and returns
	the amino-acid frequencies of a window centered at each potential codon
	position.  The positions are forward and reverse interlaced and so the
	rows follow the pattern:
		+1 -1 +2 -2 +3 -3 +4 -4 +5 -5 +6 -6 +7 -7 etc

	The row consists of 41 columns, where the first column is the GC content
	and the other 40 are the amino-acid frequencies, and are interlaced by
	those before and those after the codon:
		A(before) A(after) C(before) C(after) D(before) D(after) etc
	'''
	if type(dna) is not str:
		dna = dna.decode()

	translate = Translate()
	gc = gc_content(dna) 

	# get the aminoacid frequency window
	args = lambda: None
	args.window = 120
	half = int(args.window / 2)
	for i in range(0, len(dna)-2, 3):
		for f in [0,1,2]:
			n = (i+f)
			window = dna[max(0+f, n-57) : n+60]
			freqs = translate.frequencies(window)
			'''
			befor = dna[ max(0+f,n-48) : n+3   ]
			after = dna[         n     : n+48  ]
			bef = translate.frequencies(befor)
			aft = translate.frequencies(after)
			'''
			row = [gc]
			#row = [i+f+1, gc]
			for aa in translate.amino_acids:
				#row.append(bef.get(aa,0))
				#row.append(aft.get(aa,0))
				row.append(freqs.get(aa,0))
			yield  row
			freqs = translate.frequencies(window, rev=True)
			#bef = translate.frequencies(befor, rev=True)
			#aft = translate.frequencies(after, rev=True)
			row = [gc]
			#row = [-(i+f+1), gc]
			for aa in translate.amino_acids:
				#row.append(bef.get(aa,0))
				#row.append(aft.get(aa,0))
				row.append(freqs.get(aa,0))
			yield row



if __name__ == '__main__':
	usage = 'make_train.py [-opt1, [-opt2, ...]] infile'
	parser = argparse.ArgumentParser(description='', formatter_class=RawTextHelpFormatter, usage=usage)
	parser.add_argument('infile', type=is_valid_file, help='input file in genbank format')
	parser.add_argument('-o', '--outfile', action="store", default=sys.stdout, type=argparse.FileType('w'), help='where to write the output [stdout]')
	parser.add_argument('-w', '--window', action="store", type=int, default=120,  help='The size of the window')
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





