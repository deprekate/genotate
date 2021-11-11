#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import io
import sys
import re
from math import log
import gzip
import random
import argparse
from argparse import RawTextHelpFormatter
from collections import Counter
import pathlib

#sign = lambda x: (1, -1)[x<0]

#import faulthandler
#sys.settrace

from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE,SIG_DFL)

#from genotate.windows import get_windows
#from genotate.make_train import get_windows

from read_genbank import GenbankFile
try:
	from aminoacid import Translate
except:
	from genotate.aminoacid import Translate


def is_valid_file(x):
	if not os.path.exists(x):
		raise argparse.ArgumentTypeError("{0} does not exist".format(x))
	return x

def same_frame(a,b):
	return (a)%3 == (b-2)%3

def nint(x):
	return int(x.replace('<','').replace('>',''))

def gc_content(seq):
	a = seq.count('a')
	c = seq.count('c')
	g = seq.count('g')
	t = seq.count('t')
	tot = a+c+g+t if a+c+g+t else 1
	return (c+g) / tot

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



def parse_genbank(infile):
	genbank = GenbankFile(infile)

	# label the positions
	for locus in genbank.values():
		positions = dict()
		for feature in locus.features(include=['CDS']):
			for i,*_ in feature.codon_locations():
				# do the other 5 frames
				for sign,offset in [(+1,1), (+1,2), (-1,1), (-1,2), (-1,0)]:
					pos = sign * (i + offset) * feature.direction
					if pos not in positions:
						positions[pos] = 0
				# do the current frame
				sign,offset = (+1,0)
				pos = sign * (i + offset) * feature.direction
				positions[pos] = 1

		# label the windows
		windows = get_windows(locus.dna)
		for i, window in enumerate(windows, start=1):
			pos = -((i+1)//2) if (i+1)%2 else ((i+1)//2)
			yield [positions.get(pos, 2)] + [rround(w, 5) for w in window]
			#yield [positions.get(pos, 2)] + window

def rev_comp(seq):
	seq_dict = {'a':'t','t':'a','g':'c','c':'g',
                'n':'n',
                'r':'y','y':'r','s':'s','w':'w','k':'m','m':'k',
                'b':'v','v':'b','d':'h','h':'d'}
	return "".join([seq_dict[base] for base in reversed(seq)])

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

						
	

def single_window(dna, n, strand, translate):
	'''
	get ONE window of 117 bases centered at the CODon
				.....COD.....   => translate => count aminoacids => [1,2,...,19,20]
	'''
	row = []
	#translate = Translate()
	window = dna[ max( n%3 , n-57) : n+60]

	#translate.image(window, strand)

	#row.extend([window])	
	#row.extend([strand])	
	#row.extend([translate.seq(window, strand)])	
	#row.extend([gc_content(window)])	
	#row.extend(nucl_freq(window, strand))
	#row.extend(gc_fp(window, strand))
	row.extend(nucl_fp(window, strand))
	freqs = translate.frequencies(window, strand)
	for aa in translate.amino_acids:
		row.append(freqs.get(aa,0.0))
	#row.extend(translate.codings(window, strand))
	#row.extend(translate.dicodings(window, strand))
	#row.extend(translate.tricodings(window, strand))
	row.extend(translate.dimers(window, strand))
	#row.extend(translate.dipeps(window, strand))
	#row.extend(translate.trimers(window, strand))
	#row.extend(translate.structure(window, strand))
	#row.extend( [translate.array(window, strand)] )
	#ents = translate.codon_entropy(window, strand)
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

	translate = Translate()
	gc = gc_content(dna) 

	# get the aminoacid frequency window
	#args = lambda: None
	for n in range(0, len(dna)-2, 3):
		for f in [0,1,2]:
			yield [gc] + single_window(dna, n+f, +1, translate)
			yield [gc] + single_window(dna, n+f, -1, translate)
			#yield [gc] + single_window(dna, n+f, +1) +  single_window(dna, n+f, -1 )
			#yield [gc] + double_window(dna, n+f, +1)
			#yield [gc] + double_window(dna, n+f, -1 )
			#yield [gc] + glob_window(dna, n+f, +1 )
			#yield [gc] + glob_window(dna, n+f, -1 )
			#yield [gc] + dglob_window(dna, n+f, +1 )
			#yield [gc] + dglob_window(dna, n+f, -1 )
	#for n in range(0, len(dna)-2, 3):
	#	for f in [0,1,2]:
	#		yield [gc] + single_window(dna, n+f, -1)

def rround(item, n):
	try:
		return round(item, n)
	except:
		try:
			return item.decode()
		except:
			return item


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


	# print the column header and quit
	if args.labels:
		translate = Translate()
		sys.stdout.write('\t'.join(['TYPE','ID', 'GC']))
		sys.stdout.write('\t')
		sys.stdout.write('\t'.join(['1a','1c','1g','1t', '2a','2c','2g','2t', '3a','3c','3g','3t']))
		sys.stdout.write('\t')
		sys.stdout.write('\t'.join([aa for aa in translate.amino_acids]))
		sys.stdout.write('\t')
		sys.stdout.write('\t'.join([aa+bb for aa in translate.amino_acids for bb in translate.amino_acids]))
		sys.stdout.write('\n')

	#faulthandler.enable()
	if os.path.isdir(args.infile):
		#raise NotImplementedError('Running on multiple files in a directory')
		for infile in os.listdir(args.infile):
			for row in parse_genbank(os.path.join(args.infile, infile)):
				args.outfile.write('\t'.join(map(str,row)))
				args.outfile.write('\n')
	else:
		if pathlib.Path(args.infile.replace('.gz','')).suffix in ['.gb', '.gbk']:
			rows = parse_genbank(args.infile)
		else:
			s = list(read_fasta(args.infile).values())[0]
			rows = get_windows(s)
		
		for row in rows:
			args.outfile.write('\t'.join(map(str,[rround(item, 5) for item in row])))
			#args.outfile.write('\t'.join(map(str,row)))
			args.outfile.write('\n')
			pass





