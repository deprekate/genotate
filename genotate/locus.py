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
from itertools import zip_longest, chain
from itertools import cycle

from genbank.locus import Locus
from genotate.feature import Feature

import genotate.codons as cd

from itertools import tee, islice

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



class Locus(Locus, feature=Feature):
	def init(self, args):
		pass
		#self.locations = cd.Locations(self.dna)

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
					_last.tags['merged'] = _last.pairs + _curr.pairs
					_last.pairs = (( str(_last.left()),str(_curr.right()) ),)
					#_last.tags['seq'] = seq
					_last.tags['other'] = _last.pairs
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
					_last.tags['merged'] = _last.pairs + _curr.pairs
					_last.pairs = (( str(_last.left()),str(_next.right()) ),)
					#_last.tags['seq'] = seq
					#_last.tags['pstop'] = (1-self.pstop()) ** (len(seq)/3)
					#_last.tags['pstopl'] = (1-self.pstop(rev_comp(seq))) ** (len(seq)/3)
					#_last.tags['merged'] = 'true'
					_curr.tags['embedded'] = 'true'
					self[_last] = True
					_last,_curr = _curr,_last
					continue
			elif _curr.strand > 0:
				pass
			elif _curr.strand < 0:
				pass
				'''
				if abs(_curr.stop_distance()) > 100 and _curr.nearest_stop() < _last.right():
					del self[_last]
					del self[_curr]
					_last.pairs = _last.pairs + _curr.pairs
					_last.tags['joined'] = 'true'
					self[_last] = True
					_curr = _next
					continue
				'''
					
			_last = _curr
			_curr = _next

