import sys
from math import log
from math import exp

from .functions import *


class CDS:
	def __init__(self, start, stop, frame, parent): 
		self.type   = 'CDS'
		self.start  = start
		self.stop   = stop
		self.frame  = frame
		self.parent = parent
		self.dna    = self.dna()
		self.rbs    = self.rbs()
		self.length = self.length()
		self.scores = []

	def dna(self):
		if self.frame > 0:
			return          self.parent.seq(self.left(), self.right())
		else:
			return rev_comp(self.parent.seq(self.left(), self.right()))


	def score(self):
		x = 0
		for a in self.scores:
			x += log( a / (1-a)) if 1>a>0 else 0
		return ave(self.scores), round( 1 / ( 1 + exp( -x / len(self.scores))), 3 )

	def length(self):
		return len(self.dna)

	def rbs(self):
		if self.frame > 0:
			return          self.parent.seq(self.left()-19, self.left()-1)
		else:
			return rev_comp(self.parent.seq(self.right()+1, self.right()+19))

	def direction(self):
		if self.frame > 0:
			return '+'
		else:
			return '-'

	def begin(self):
		if self.frame > 0:
			c = '' if self.start_codon() in self.parent.start_codons else '<'
			return c + str(self.start)
		else:
			c = '' if self.start_codon() in self.parent.start_codons else '>'
			return c + str(self.start+2)

	def end(self):
		if self.frame > 0:
			c = '' if self.stop_codon() in self.parent.stop_codons else '>'
			return c + str(self.stop+2)
		else:
			c = '' if self.stop_codon() in self.parent.stop_codons else '<'
			return c + str(self.stop)

	def left(self):
		if self.frame > 0:
			return self.start
		else:
			return self.stop

	def right(self):
		if self.frame > 0:
			return self.stop + 2
		else:
			return self.start + 2

	def codon_counts(self):
		codons = dict()
		for i in range(0, self.length(), 3):
			codon = self.dna[i:i+3]
			codons[codon] = codons.get(codon, 0) + 1
		return codons
		
	def amino_acids(self):
		"""Calculate the amino acid frequency"""
		aa = []
		for i in range(0, self.length(), 3):
			aa.append(self.parent.translate_codon[self.dna[i:i+3]])
		return "".join(aa)

	def amino_acid_frequencies(self):
		amino_acids = self.amino_acids()
		counts = dict()
		total = 0
		for aa in list('ACDEFGHIKLMNPQRSTVWY#+*'):
			counts[aa] = amino_acids.count(aa)
			total += amino_acids.count(aa)
		for aa in list('ACDEFGHIKLMNPQRSTVWY#+*'):
			counts[aa] = counts[aa] / total
		return counts

	def amino_acid_count(self, aa):
		return self.amino_acids().count(aa) 
			
	def amino_acid_frequency(self, aa):
		return self.amino_acid_count(aa) / len(self.amino_acids)

	def start_codon(self):
		return self.dna[0:3]

	def stop_codon(self):
		return self.dna[-3:]

	def has_start(self):
		return self.start_codon() in self.parent.start_codons
		
	def has_stop(self):
		return self.stop_codon() in self.parent.stop_codons

	def __repr__(self):
		"""Compute the string representation of the orf"""
		return "%s(%s,%s,%s)" % (
			self.__class__.__name__,
			repr(self.start),
			repr(self.stop),
			repr(self.frame)
			#repr(self.score)
		)

	def __eq__(self, other):
		"""Override the default Equals behavior"""
		if isinstance(other, self.__class__):
			return self.__dict__ == other.__dict__
		return NotImplemented

	def __ne__(self, other):
		"""Define a non-equality test"""
		if isinstance(other, self.__class__):
			return not self.__eq__(other)
		return NotImplemented

	def __hash__(self):
		"""Override the default hash behavior (that returns the id or the object)"""
		return hash(tuple(sorted(self.__dict__.items())))

