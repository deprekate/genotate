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
from itertools import zip_longest, chain, tee, islice, tee
from itertools import cycle
from copy import deepcopy
from textwrap import wrap
from math import log

from genbank.locus import Locus
from genotate.feature import Feature
from genotate.functions import rev_comp

import numpy as np

def previous_and_next(some_iterable):
	prevs, items, nexts = tee(some_iterable, 3)
	prevs = chain([None], prevs)
	nexts = chain(islice(nexts, 1, None), [None])
	return zip(prevs, items, nexts)

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def count(dq, item):
	return sum(elem == item for elem in dq)
def nint(x):
	return int(x.replace('<','').replace('>',''))

class Locus(Locus, feature=Feature):
	def init(self, args):
		#self._rbs = score_rbs.ScoreXlationInit()
		self.stops = ['taa','tga','tag']
		pass
	def has_stop(self, dna):
		for i in range(0, len(dna), 3):
			if dna[i:i+3] in self.stops:
				return True
		return False

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
	
	def join(self):	
		lefts = list()
		rights = list()
		for feature in sorted(self):
			if feature.left() <= 3:
				if feature.strand > 0 and feature.start_codon()[:2] != 'tg':
					lefts.append(feature)
				elif feature.strand < 0 and feature.stop_codon() not in self.stops:
					lefts.append(feature)
			elif feature.right() > self.length() - 6:
				rights.append(feature)
		pass

	def merge(self):	
		_last = _curr = None
		#for _, _, _next in previous_and_next(sorted(self)):
		for _next in chain(sorted(self.features(include='CDS')), [None]):
			# THIS JUST MKES SURE THE FEATURE LOCATIONS ARE IN THE SAME FRAME
			#for i in _curr.base_locations():
			#	_curr.dna += self.dna[ i-1 : i ]
			if _curr is None or _next is None: # or (_last.type != 'CDS') or (_curr.type != 'CDS'):
				pass
			#elif _curr.strand != _last.strand:
			#	pass
			#elif _last.frame('right') == _curr.frame('left'):
			elif _curr.frame() == _next.frame():
				# THIS MERGES ADJACENT FRAMES
				seq = self.seq(_curr.right()-30 , _next.left()+30, _next.strand)
				if not self.has_stop(seq):
					del self[_curr]
					del self[_next]
					#_curr.tags['note'] = ['"merged:' + str(_curr.pairs) + str(_next.pairs) + '"' ]
					_curr.pairs = ( (_curr.pairs[0][0] , _next.pairs[-1][-1]), )
					self[_curr] = True
					#_curr = _next
					#continue
					_curr,_next = _last,_curr
			elif _last is None:
				pass
			elif _last.frame() == _next.frame():
				# THIS MERGES FRAMES BROKEN BY AN EMBEDDED GENE
				seq = self.seq(_last.right()-30 , _next.left()+30, _next.strand)
				if not self.has_stop(seq):
					del self[_last]
					del self[_next]
					#_last.tags['note'] = ['"merged:' + str(_last.pairs) + str(_next.pairs) + '"' ]
					_last.pairs = ((_last.pairs[0][0] , _next.pairs[-1][-1]), )
					#_curr.tags['embedded'] = ['true']
					self[_last] = True
					#_last,_curr = _curr,_last
					_next = _last
			_last = _curr
			_curr = _next

	def split(self):
		#stops = ['taa','tga','tag']
		for feature in sorted(self.features(include='CDS')):
			stop_locations = [feature.left()-3]
			for codon,locs in zip(feature.codons(), feature.codon_locations()):
				if codon in self.stops:
					stop_locations.append(min(locs))
			stop_locations.append(feature.right())
			del self[feature]
			for left,right in pairwise(sorted(set(stop_locations))):
				if right-left >= 60:
					if feature.strand > 0:
						pairs = [[left+4,right+3]]
					else:
						pairs = [[left+1,right+0]]
					pairs = tuple([tuple(map(str,pair)) for pair in pairs])
					self.add_feature('CDS', feature.strand, pairs, tags=feature.tags)

	def adjust(self):
		#stops = ['taa','tga','tag']
		seen = dict()
		for feature in sorted(self.features(include='CDS')):
			if feature.stop_codon() not in self.stops:
				del self[feature]
				# I NEED TO ADD SOME LOGIC HERE TO LIMIT THE MOVEMENT TO NOT
				# GREATER THAN THE LENGTH OF THE FEATURES CURRENT LENGTH

				if feature.strand > 0:
					#left  = self.last(feature.right(), self.stops, feature.strand)
					right = self.next(feature.right(), self.stops, feature.strand)
					#left  = left  if left  else 0
					right = right if right else self.length()-3
					feature.set_right(right+3)
				else:
					left  = self.next(feature.left(), self.stops, feature.strand)
					#right = self.last(feature.left(), self.stops, feature.strand)
					left = left+1 if left else '<1'
					feature.set_left(left)
				self[feature] = True
			# THIS REMOVES DUPLICATE GENES WITH SAME STOP CODON
			if feature.end() in seen:
				other = seen[feature.end()]
				if feature.length() > other.length():
					del self[other]
				else:
					del self[feature]
					feature = other
			seen[feature.end()] = feature

	def count_starts(self):
		counts = dict()
		for feature in self:
			codon = feature.seq()[0:3]
			counts[codon] = counts.setdefault(codon,0) + 1
		return counts

	def detect_stops(self):
		n = 10
		counts = {stop : 0 for stop in self.stops}
		for _last, _curr, _next in previous_and_next(sorted(self.features(include='CDS'))):
			if _last and _last.frame() == _curr.frame():
				seq = self.seq(_last.right(), _curr.left(), _last.frame())
				codons = wrap(seq, 3)
				if len(seq) < 300 and sum(['taa' in codons, 'tag' in codons, 'tga' in codons]) == 1:
					for stop in self.stops:
						counts[stop] += stop in codons #codons.count(stop)

			acids = _curr.translation()[  :-n]
			# only count if one type of stop codon is present
			if sum(['#' in acids, '+' in acids, '*' in acids]) == 1:
				counts['taa'] += "#" in acids #acids.count('#')
				counts['tag'] += '+' in acids #acids.count('+')
				counts['tga'] += '*' in acids # acids.count('*')
		stops = self.stops
		for stop in self.stops:
			counts[stop] /= len(self) if len(self) else 1
			if len(self) > 20 and counts[stop] > 0.20 :
				stops.remove(stop)
		stops = keys(counts) if len(stops) == 1 else stops
		return stops

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
