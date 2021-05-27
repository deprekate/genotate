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
from pathlib import Path


def is_valid_file(x):
	if not os.path.exists(x):
		raise argparse.ArgumentTypeError("{0} does not exist".format(x))
	return x

def same_frame(a,b):
	return (a)%3 == (b-2)%3

def nint(x):
	return int(x.replace('<','').replace('>',''))

def read_genbank(infile):
	length = num = total = 0
	forward = []
	reverse = []
	with open(infile) as fp:
		for line in fp:
			if line.startswith('LOCUS'):
				length = int(line.split()[2])
				forward = [0] * length
				reverse = [0] * length
			elif line.startswith('     CDS '):
				pairs = [pair.split('..') for pair in re.findall(r"<*\d+\.\.>*\d+", line)]
				# this is for features that continue on next line
				if line.rstrip().endswith(','):
					pairs.extend([pair.split('..') for pair in re.findall(r"<*\d+\.\.>*\d+", next(fp))])
				# this is for weird malformed features
				if ',1)' in line:
					pairs.append(['1','1'])
				# loop over the feature recording its location
				remainder = 0
				for pair in pairs:
					left,right = map(nint, pair)
					if '<' in pair[0]:
						left = left + ((int(pairs[-1][-1])) - left -2) % 3
					for i in range(left-remainder,right-1,3):
						if 'complement' in line:
							reverse[i] = 1
						else:
							forward[i] = 1
						remainder = right-2 - i
				if remainder and ">" not in pair[1]:
					raise ValueError("Out of frame: ( %s , %s )" % tuple(pair))
	for i in range(0, length-2, 3):
		total = sum(forward[i:i+3]) + sum(reverse[i:i+3])
		if total > 1:
			num += (total-1)
	return (3*num, length)



if __name__ == '__main__':
	usage = 'make_train.py [-opt1, [-opt2, ...]] infile'
	parser = argparse.ArgumentParser(description='', formatter_class=RawTextHelpFormatter, usage=usage)
	parser.add_argument('infile', type=is_valid_file, help='input file in genbank format')
	parser.add_argument('-o', '--outfile', action="store", default=sys.stdout, type=argparse.FileType('w'), help='where to write output [stdout]')
	#parser.add_argument('-w', '--window', action="store", type=int, default=120,  help='The size of the window')
	parser.add_argument('-l', '--labels', action="store_true", help=argparse.SUPPRESS)
	parser.add_argument('--ids', action="store", help=argparse.SUPPRESS)
	args = parser.parse_args()

	o, l = read_genbank(args.infile)
	print(Path(args.infile).stem, o, l, round(o/l, 3), sep='\t')





