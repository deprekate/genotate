import sys
import itertools

from .orf import CDS
from .functions import *

class Features(list):
	"""The class holding the orfs"""
	def __init__(self, n=0, **kwargs):
		self.__dict__.update(kwargs)

		self.n = n
		self.dna = None
		self.cds = dict()
		self.feature_at = dict()

		nucs = ['t', 'c', 'a', 'g']
		codons = [a+b+c for a in nucs for b in nucs for c in nucs]
		amino_acids = 'FFLLSSSSYY#+CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'
		self.translate_codon = dict(zip(codons, amino_acids))

		self.start_codons = ['atg', 'gtg', 'ttg']
		self.stop_codons  = ['tga', 'tag', 'taa']
		self.min_orf_len = 87


	def seq(self, a, b):
			return self.dna[ a-1 : b ]

	def add_orf(self, start, stop, frame, seq, rbs, scores):
		""" Adds an orf to the factory"""
		if len(seq) < self.min_orf_len: return

		o = CDS(start, stop, frame, self)
		o.scores = scores
		#print(o, o.scores)

		orfs = self.cds
		if stop not in orfs:
			orfs[stop] = dict()
			orfs[stop][start] = o
		elif start not in orfs[stop]:
			orfs[stop][start] = o
		else:
			raise ValueError("orf already defined")
		self.add_feature(o)

	def add_feature(self, feature):
		self.append(feature)

	def iter_features(self, type_ = None):
		for feature in self:
			if feature.type != type_:
				continue
			yield feature

	def iter_orfs(self, kind=None):
		orfs = self.cds
		if not kind:
			for stop in orfs:
				for start in orfs[stop]:
					yield orfs[stop][start]
		elif kind == 'in' or kind == 'longest':
			for stop in orfs.keys():
				keylist = list(orfs[stop].keys())
				if(orfs[stop][keylist[0]].frame > 0):
					keylist.sort()
				else:
					keylist.sort(reverse=True)
				if kind == 'longest':
					keylist = keylist[:1]
				yield (orfs[stop][start] for start in keylist)
		elif kind == 'out':
			for stop in orfs.keys():
				keylist = list(orfs[stop].keys())
				if(orfs[stop][keylist[0]].frame > 0):
					keylist.sort(reverse=True)
				else:
					keylist.sort()
				yield (orfs[stop][start] for start in keylist)
		elif kind == 'half':
			for stop in orfs.keys():
				keylist = list(orfs[stop].keys())
				if(orfs[stop][keylist[0]].frame > 0):
					keylist.sort()
				else:
					keylist.sort(reverse=True)
				keylist = keylist[ : -(-len(keylist)//2)]
				yield (orfs[stop][start] for start in keylist)

	def get_orf(self, start, stop):
		orfs = self.cds
		if stop in orfs:
			if not start:
				return orfs[stop]
			elif start in orfs[stop]:
				return orfs[stop][start]
			else:
				raise ValueError("orf with start codon not found")
		else:
			raise ValueError(" orf with stop codon not found")

	def get_orfs(self, stop):
		orfs = self.cds
		if stop in orfs:
			return orfs[stop]
		else:
			raise ValueError(" orf with stop codon not found")

	def get_feature(self, left, right):
		return self.feature_at.get( (left, right) , None )

	def contig_length(self):
		return len(self.dna)
	
	def end(self, frame):
		return self.contig_length() - ((self.contig_length() - (frame-1))%3)

	def num_orfs(self, kind='all'):
		count = 0
		if kind == 'all':
			for orf in self.iter_orfs():
				count += 1
		else:
			count = len(self.cds)
		return count

	def parse_contig(self, id, dna, preds):
		self.id = id
		self.dna = dna
	
		# The dicts that will hold the start and stop codons
		stops = {1:0, 2:0, 3:0, -1:1, -2:2, -3:3}
		starts = {1:[], 2:[], 3:[], -1:[], -2:[], -3:[]}
		counts = {1:[], 2:[], 3:[], -1:[], -2:[], -3:[]}
		bank = {1:[], 2:[], 3:[], -1:[], -2:[], -3:[]}
		if dna[0:3] not in self.start_codons:
			starts[1].append(1)
		if dna[1:4] not in self.start_codons:
			starts[2].append(2)
		if dna[2:5] not in self.start_codons:
			starts[3].append(3)

		# Reset iterator and find all the open reading frames
		states = itertools.cycle([1, 2, 3])
		for i in range(1, (len(dna)-1)):
			frame = next(states)
			codon = dna[i-1:i+2]
			#print(i, codon, preds[2*(i-1)], preds[2*(i-1)+1], counts)
			counts[ frame].append( preds[2*(i-1)  ] )
			counts[-frame].append( preds[2*(i-1)+1] )
			if codon in self.start_codons:
				if not starts[frame]:
					counts[frame] = []
				starts[frame].append(i)
			elif rev_comp(codon) in self.start_codons:
				starts[-frame].append(i+2)
				counts[-frame] = bank[-frame]
			elif codon in self.stop_codons:
				stop = i+2
				for start in reversed(starts[frame]):
					seq = dna[start-1:stop]
					rbs = dna[start-21:start]
					self.add_orf(start, stop-2, frame, seq, rbs, counts[frame])
		
				starts[frame] = []
				stops[frame] = stop
				counts[frame] = []
				bank[frame] = []
			elif rev_comp(codon) in self.stop_codons:
				stop = stops[-frame]
				for start in starts[-frame]:
					seq = rev_comp(dna[max(0,stop-1):start])
					rbs = rev_comp(dna[start:start+21])
					self.add_orf(start-2, stop, -frame, seq, rbs, counts[-frame])
		
				starts[-frame] = []
				stops[-frame] = i
				counts[-frame] = []
				bank[-frame] = []
		# Add in any fragment ORFs at the end of the genome
		for frame in [1, 2, 3]:
			for start in reversed(starts[frame]):
				stop = self.end(frame)
				seq = dna[max(0,start-1):stop]
				rbs = dna[start-21:start]
				self.add_orf(start, stop-2, frame, seq, rbs, counts[frame])
			start = self.end(frame)
			if rev_comp(dna[start-3:start]) not in self.start_codons:
				starts[-frame].append(self.end(frame))	
			for start in starts[-frame]:
				stop = stops[-frame]
				seq = rev_comp(dna[max(0,stop-1):start])
				rbs = rev_comp(dna[start:start+21])
				self.add_orf(start-2, stop, -frame, seq, rbs, counts[-frame])

	def first_start(self, stop):
		if stop in self:
			list = sorted(list(self[stop].keys()))
			if(list[0] < stop):
				return list[0]
			else:
				return list[-1]


