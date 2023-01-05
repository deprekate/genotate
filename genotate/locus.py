#!/usr/bin/env python3

import os
import io
import sys
import gzip
import re
import argparse
import tempfile
from collections import Counter
from argparse import RawTextHelpFormatter
from itertools import zip_longest, chain, tee, islice
from itertools import cycle

from genbank.locus import Locus
from genotate.feature import Feature

import numpy as np

def previous_and_next(some_iterable):
	prevs, items, nexts = tee(some_iterable, 3)
	prevs = chain([None], prevs)
	nexts = chain(islice(nexts, 1, None), [None])
	return zip(prevs, items, nexts)

def count(dq, item):
	return sum(elem == item for elem in dq)
def nint(x):
	return int(x.replace('<','').replace('>',''))

def find_median(sorted_list):
	indices = []
	list_size = len(sorted_list)
	median = 0
	if list_size % 2 == 0:
		indices.append(int(list_size / 2) - 1)  # -1 because index starts from 0
		indices.append(int(list_size / 2))
		median = (sorted_list[indices[0]] + sorted_list[indices[1]]) / 2
		pass
	else:
		indices.append(int(list_size / 2))
		median = sorted_list[indices[0]]
		pass
	return median, indices
	pass

def has_outlier(unsorted_list):
	if min(unsorted_list) < (np.mean(unsorted_list) - 2*np.std(unsorted_list)):
		return True
	else:
		return False
	sorted_list = sorted(unsorted_list)
	median, median_indices = find_median(sorted_list)
	q25, q25_indices = find_median(sorted_list[:median_indices[0]])
	q75, q75_indices = find_median(sorted_list[median_indices[-1] + 1:])
	iqr = q75 - q25
	lower = q25 - (iqr * 1.5)
	if sorted_list[0] < lower:
		return True
	else:
		return False

def has_stop(dna, strand):
	codons = ['taa','tag','tga'] if strand > 0 else ['tta','cta','tca']
	for i in range(0, len(dna), 3):
		if dna[i:i+3] in codons:
			return True
	return False

def previous_and_next(some_iterable):
	prevs, items, nexts = tee(some_iterable, 3)
	prevs = chain([None], prevs)
	nexts = chain(islice(nexts, 1, None), [None])
	return zip(prevs, items, nexts)

class Locus(Locus, feature=Feature):

	def init(self, args):
		#self._rbs = score_rbs.ScoreXlationInit()
		self.stops = ['taa','tga','tag']
		pass

	def score_rbs(self, rbs):
		return self._rbs.score_init_rbs(rbs,20)[0]


	def pstart(self):
		return self.pcodon('atg') + self.pcodon('gtg') + self.pcodon('ttg')

	def pstop(self, seq=None):
		if seq is None:
			return self.pcodon('taa') + self.pcodon('tag') + self.pcodon('tga')
		else:
			length = len(seq)
			pa = seq.count('a') / length
			pc = seq.count('c') / length
			pg = seq.count('g') / length
			pt = seq.count('t') / length
			return pt*pa*pa + pt*pa*pg + pt*pg*pa

	def merge(self):	
		_last = _curr = None
		for _, _, _next in previous_and_next(sorted(self)):
			# THIS JUST MKES SURE THE FEATURE LOCATIONS ARE IN THE SAME FRAME
			#for i in _curr.base_locations():
			#	_curr.dna += self.dna[ i-1 : i ]
			if _last is None or (_last.type != 'CDS') or (_curr.type != 'CDS'):
				pass
			elif _curr.strand != _last.strand:
				pass
			elif _last.frame('right') == _curr.frame('left'):
				# THIS MERGES ADJACENT FRAMES
				seq = self.seq(_last.right()-30 , _curr.left()+32)
				if not has_stop(seq, _last.strand):
					del self[_last]
					del self[_curr]
					_last.tags['note'] = ['"merged:' + str(_last.pairs) + str(_curr.pairs) + '"' ]
					_last.pairs = ( (_last.pairs[0][0] , _curr.pairs[-1][-1]), )
					self[_last] = True
					_curr = _next
					continue
			elif _next is None:
				pass
			elif _last.frame('right') == _next.frame('left'):
				# THIS MERGES FRAMES BROKEN BY AN EMBEDDED GENE
				seq = self.seq(_last.right()-30 , _next.left()+32)
				if not has_stop(seq, _last.strand):
					del self[_last]
					del self[_next]
					_last.tags['note'] = ['"merged:' + str(_last.pairs) + str(_next.pairs) + '"' ]
					_last.pairs = ((_last.pairs[0][0] , _next.pairs[-1][-1]), )
					_curr.tags['embedded'] = ['true']
					self[_last] = True
					_last,_curr = _curr,_last
					continue
			elif _curr.strand > 0:
				pass
			elif _curr.strand < 0:
				'''
				if abs(_curr.stop_distance()) > 100 and _curr.nearest_stop() < _last.right():
					del self[_last]
					del self[_curr]
					_last.pairs = _last.pairs + _curr.pairs
					_last.tags['joined'] = ['true']
					self[_last] = True
					_curr = _next
					continue
				'''
				pass
			_last = _curr
			_curr = _next

	def check_sequencing_error(self):
		for _last, _curr, _ in previous_and_next(sorted(self)):
			if _last is None or (_last.type != 'CDS') or (_curr.type != 'CDS'):
				pass
			elif _curr.strand != _last.strand:
				pass
			elif _curr.strand > 0:
				print(_last.right(), _curr.left(), _curr.start_distance())
			elif _curr.strand < 0:
				pass

	def adjust_ends(self, starts, stops):
		pass
		for feature in self:
			feature.adjust_stop()

	def skew(self, nucs):
		windowsize = stepsize = 100 #int(len(self.sequence) / 1000)
		(nuc_1,nuc_2) = nucs
		
		cumulative = 0
		cm_list = []
		i = windowsize // 2
		for _ in range(len(self.seq()) // stepsize):
			if i < self.length():
				a = self.seq(i - windowsize//2 , i + windowsize//2 ).count(nuc_1)
				b = self.seq(i - windowsize//2 , i + windowsize//2 ).count(nuc_2)
				s = (a - b) / (a + b) if (a + b) else 0
				cumulative = cumulative + s
				cm_list.append(cumulative)
				i = i + stepsize
		slopes = []
		for i in range(len(cm_list)):
			win = cm_list[max(i-5,0):i+5]
			m,b = np.polyfit(list(range(len(win))),win, 1)
			slopes.append(m)
		slopes.append(m)
		return slopes

	def get_windows(self):
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
	
		gc = self.gc_content() 
		dna = self.gc_content()
	
		at_skew = self.skew('at')
		gc_skew = self.skew('gc')
		# get the aminoacid frequency window
		#args = lambda: None
		forward = [0]*48 + [0]*self.length() + [0]*50
		reverse = [0]*48 + [0]*self.length() + [0]*50
		for i,base in enumerate(self.seq()):
			if base in 'acgt':
				forward[i+48] = ((ord(base) >> 1) & 3) + 1
				reverse[i+48] = ((forward[i+48] - 3) % 4) + 1
	
		for n in range(0, self.length()-2, 3):
			for f in [0,1,2]:
				yield [ gc,  at_skew[n//100],  gc_skew[n//100] ] , forward[n+f : n+f+99 ]
				yield [ gc, -at_skew[n//100], -gc_skew[n//100] ] , reverse[n+f : n+f+99 ][::-1]
				#
				#yield [str(gc), str( at_skew[n//100]), str( gc_skew[n//100]) ] + single_window(dna, n+f, +1)
				#yield [str(gc), str(-at_skew[n//100]), str(-gc_skew[n//100]) ] + single_window(dna, n+f, -1)

	def get_labeled_windows(self):
		# label the positions
		positions = dict()
		for feature in self.features(include=['CDS']):
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

		# label the windows
		windows = self.get_windows()
		for i, (data, window) in enumerate(windows, start=1):
			pos = -((i+1)//2) if (i+1)%2 else ((i+1)//2)
			yield [positions.get(pos, 2)] , data , window #[rround(w, 5) for w in window]
