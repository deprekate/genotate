#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import io
import sys

sys.path.pop(0)

from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE,SIG_DFL)

import faulthandler; faulthandler.enable()
sys.settrace

from read_genbank import GenbankFile
from genotate.windows import get_windows as cgw
from genotate.pywindows import get_windows as pgw


def rround(item, n):
	try:
		return round(item, n)
	except:
		try:
			return item.decode()
		except:
			return item


genbank = GenbankFile(sys.argv[1])

for name,locus in genbank.items():
	cwindows = cgw(locus.dna)
	pwindows = pgw(locus.dna)
	for i, windows in enumerate(zip(cwindows, pwindows), start=1):
		#print(windows[0]) ; print(windows[1])
		pw = [rround(item, 5) for item in windows[0] ]
		cw = [rround(item, 5) for item in windows[1] ]
		#print(i)
		#print(pw) ; print(cw)
		assert pw == cw, 'Error: non match at ' + str(i) + '\n%\t' + '\t'.join([str(p+1)+n for p in range(3) for n in list('acgt')]) + '\t'.join(list('#*+ACDEFGHIKLMNPQRSTVWY')) + '\n' + '\t'.join(map(str, pw)) + '\n' + '\t'.join(map(str, cw))
		pos = -((i+1)//2) if (i+1)%2 else ((i+1)//2)
		#print( [rround(w, 5) for w in window] )
		#yield [positions.get(pos, 2)] + window

print('okay')
