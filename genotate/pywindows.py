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

#try:
#	from aminoacid import Translate
#except:
#	from genotate.aminoacid import Translate

from read_genbank import Translate
translate = Translate()


def gc_content(seq):
	a = seq.count('a')
	c = seq.count('c')
	g = seq.count('g')
	t = seq.count('t')
	tot = a+c+g+t if a+c+g+t else 1
	return (c+g) / tot


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

						
	

def single_window(dna, n, strand):
	'''
	get ONE window of 117 bases centered at the CODon
				.....COD.....   => translate => count aminoacids => [1,2,...,19,20]
	'''
	global translate
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
	#row.extend(nucl_fp(window, strand))
	freqs = translate.frequencies(window, strand)
	for aa in translate.amino_acids:
		row.append(freqs.get(aa,0.0))
	#row.extend(translate.codings(window, strand))
	#row.extend(translate.dicodings(window, strand))
	#row.extend(translate.tricodings(window, strand))
	#row.extend(translate.dimers(window, strand))
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

	translate = None # Translate()
	gc = gc_content(dna) 

	# get the aminoacid frequency window
	#args = lambda: None
	for n in range(0, len(dna)-2, 3):
		for f in [0,1,2]:
			yield [gc] + single_window(dna, n+f, +1)
			yield [gc] + single_window(dna, n+f, -1)
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





