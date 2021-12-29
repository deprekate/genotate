#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import io
import sys

import pywindows

sys.path.pop(0)

from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE,SIG_DFL)

import faulthandler; faulthandler.enable()
sys.settrace

from read_genbank import GenbankFile
from genotate.windows import get_windows


def rround(item, n):
	try:
		return round(item, n)
	except:
		try:
			return item.decode()
		except:
			return item

cout = ''
pout = ''

genbank = GenbankFile("phiX174.gbk")

for name,locus in genbank.items():
	cwindows = get_windows(locus.dna)
	pwindows = pywindows.get_windows(locus.dna)
	for i, windows in enumerate(zip(cwindows, pwindows), start=1):
		pw = [rround(item, 5) for item in windows[0] ]
		cw = [rround(item, 5) for item in windows[1] ]
		print(pw)
		print(cw)
		exit()
		pos = -((i+1)//2) if (i+1)%2 else ((i+1)//2)
		cout += '\t'.join(map(str, window))
		#print( [rround(w, 5) for w in window] )
		#yield [positions.get(pos, 2)] + window

