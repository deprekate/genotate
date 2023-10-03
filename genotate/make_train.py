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

si = lambda x: (1, -1)[x<0]

#import faulthandler
#sys.settrace

from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE,SIG_DFL)

#from genotate.windows import get_windows
#from genotate.make_train import get_windows

import numpy as np
from genbank.file import File

def rround(item, n):
	try:
		return round(item, n)
	except:
		try:
			return item.decode()
		except:
			return item


def is_valid_file(x):
	if not os.path.exists(x):
		raise argparse.ArgumentTypeError("{0} does not exist".format(x))
	return x

def same_frame(a,b):
	return (a)%3 == (b-2)%3

def nint(x):
	return int(x.replace('<','').replace('>',''))

def skew(seq, nucs):
	self = lambda : none
	self.sequence = seq
	self.windowsize = self.stepsize = 100 #int(len(self.sequence) / 1000)
	(self.nuc_1,self.nuc_2) = nucs
	
	cumulative = 0
	cm_list = []
	i = int(self.windowsize / 2)
	for each in range(len(self.sequence) // self.stepsize):
		if i < len(self.sequence):
			a = self.sequence[i - int(self.windowsize / 2):i + int(self.windowsize / 2)].count(self.nuc_1)
			b = self.sequence[i - int(self.windowsize / 2):i + int(self.windowsize / 2)].count(self.nuc_2)
			s = (a - b) / (a + b) if (a + b) else 0
			cumulative = cumulative + s
			cm_list.append(cumulative)
			i = i + self.stepsize
	slopes = []
	for i in range(len(cm_list)):
		win = cm_list[max(i-5,0):i+5]
		m,b = np.polyfit(list(range(len(win))),win, 1)
		slopes.append(m)
	slopes.append(m)
	return slopes

def gc_content(seq):
	a = seq.count('a')
	c = seq.count('c')
	g = seq.count('g')
	t = seq.count('t')
	tot = a+c+g+t if a+c+g+t else 1
	return (c+g) / tot

def rev_comp(seq):
	seq_dict = {'a':'t','t':'a','g':'c','c':'g',
                'n':'n',
                'r':'y','y':'r','s':'s','w':'w','k':'m','m':'k',
                'b':'v','v':'b','d':'h','h':'d'}
	return "".join([seq_dict[base] for base in reversed(seq)])

def single_window(dna, n, strand):
	'''
	get ONE window of 117 bases centered at the CODon
				.....COD.....   => translate => count aminoacids => [1,2,...,19,20]
	'''
	row = []
	#window = dna[ max( n%3 , n-57) : n+60]
	#window = dna[ max( n%3 , n-72) : n+75]
	window = dna[ max( 0 , n-72) : n+75]

	if n-72 < 0:
		window = window.rjust(147, 'n')
	elif len(window) < 147:
		window = window.ljust(147, 'n')
		
	#row.extend([window])	
	if strand > 0:
		row.extend([window])	
	else:
		row.extend([rev_comp(window)])	
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
	#if type(dna) is not str:
	#	dna = dna.decode()

	#gc = gc_content(dna) 

	#at_skew = skew(dna, 'at')
	#gc_skew = skew(dna, 'gc')
	# get the aminoacid frequency window
	#args = lambda: None
	forward = 'n'*48 +          dna  + 'n'*50
	reverse = 'n'*48 + rev_comp(dna) + 'n'*50
	'''
	forward = [0]*48 + [0]*len(dna) + [0]*50
	reverse = [0]*48 + [0]*len(dna) + [0]*50
	for i,base in enumerate(dna):
		if base in 'acgt':
			forward[i+48] = ((ord(base) >> 1) & 3) + 1
			reverse[i+48] = ((forward[i+48] - 3) % 4) + 1
	'''
	for n in range(0, len(dna)-2, 3):
		for f in [0,1,2]:
			yield forward[n+f : n+f+99 ]
			yield rev_comp(forward[n+f : n+f+99 ])
			#yield [ gc,  at_skew[n//100],  gc_skew[n//100] ] , forward[n+f : n+f+99 ]
			#yield [ gc, -at_skew[n//100], -gc_skew[n//100] ] , reverse[n+f : n+f+99 ][::-1]
			#
			#yield [str(gc), str( at_skew[n//100]), str( gc_skew[n//100]) ] + single_window(dna, n+f, +1)
			#yield [str(gc), str(-at_skew[n//100]), str(-gc_skew[n//100]) ] + single_window(dna, n+f, -1)

def parse_genbank(infile):
	genbank = File(infile)
	# label the positions
	for locus in genbank:
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
		dna = locus.seq()
		gc = gc_content(dna) 
		at_skew = skew(dna, 'at')
		gc_skew = skew(dna, 'gc')
		forward = 'n'*48 +          dna  + 'n'*50
		reverse = 'n'*48 + rev_comp(dna) + 'n'*50
		for n in range(0, len(dna)-2, 3):
			for f in [0,1,2]:
				yield [positions.get( n+f, 2)] , [ gc,  at_skew[n//100],  gc_skew[n//100] ] , [forward[n+f : n+f+99 ]]
				yield [positions.get(-n+f, 2)] , [ gc, -at_skew[n//100], -gc_skew[n//100] ] , [reverse[n+f : n+f+99 ][::-1]]


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
		rows = parse_genbank(args.infile)
		for row in rows:
			#if len(row[-1]) < 147:
			#	continue
			#seq = row[-1].replace('t','u')
			#args.outfile.write( seq )
			#args.outfile.write('\t')
			#args.outfile.write( str(lf.fold( seq )[1]) )
			args.outfile.write('\t'.join(map(str,[rround(item, 5) for item in row])))
			#args.outfile.write('\t'.join(map(str,row)))
			args.outfile.write('\n')
			pass





