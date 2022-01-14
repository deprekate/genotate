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

import genotate.codons as cd

def is_valid_file(x):
	if not os.path.exists(x):
		raise argparse.ArgumentTypeError("{0} does not exist".format(x))
	return x

def nint(x):
    return int(x.replace('<','').replace('>',''))

def rev_comp(dna):
	a = 'acgtrykmbvdh'
	b = 'tgcayrmkvbhd'
	tab = str.maketrans(a,b)
	return dna.translate(tab)[::-1]

def has_stop(dna):
	for i in range(0, len(dna), 3):
		if dna[i:i+3] in ['taa','tag','tga']:
			return True
	return False

def mask(seq1, seq2):
    out1 = out2 = ''
    for tup in zip(seq1,seq2):
        if 'X' not in tup:
            out1 += tup[0]
            out2 += tup[1]
    return out1,out2

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

class Translate:
	def __init__(self):
		nucs = 'acgt'
		codons = [a+b+c for a in nucs for b in nucs for c in nucs]
		amino_acids = 'KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV#Y+YSSSS*CWCLFLF'
		self.translate = dict(zip(codons, amino_acids))
		ambi = """aarK aayN acrT acyT ackT acmT acbT acvT acdT achT agrR agyS atyI atmI athI 
		carQ cayH ccrP ccyP cckP ccmP ccbP ccvP ccdP cchP cgrR cgyR cgkR cgmR cgbR cgvR cgdR 
		cghR ctrL ctyL ctkL ctmL ctbL ctvL ctdL cthL garE gayD gcrA gcyA gckA gcmA gcbA gcvA 
		gcdA gchA ggrG ggyG ggkG ggmG ggbG ggvG ggdG gghG gtrV gtyV gtkV gtmV gtbV gtvV gtdV 
		gthV tar* tayY tcrS tcyS tckS tcmS tcbS tcvS tcdS tchS tgyC ttrL ttyF tra* ytaL ytgL 
		ytrL mgaR mggR mgrR"""
		for item in ambi.split():
			self.translate[item[0:3]] = item[-1]
		self.amino_acids = sorted(set(amino_acids))

	def rev_comp(self, seq):
		seq_dict = {'a':'t','t':'a','g':'c','c':'g',
					'n':'n',
					'r':'y','y':'r','s':'s','w':'w','k':'m','m':'k',
					'b':'v','v':'b','d':'h','h':'d'}
		return "".join([seq_dict[base] for base in reversed(seq)])

	def codon(self, codon):
		if len(codon) == 3:
			return self.translate.get(codon.lower(), 'X')
		else:
			return ''

	def sequence(self, seq, strand):
		aa = ''
		if strand > 0:
			for i in range(0, len(seq), 3):
				aa += self.codon(seq[i:i+3])
			return aa
		else:
			for i in range(0, len(seq), 3):
				aa += self.codon(self.rev_comp(seq[i:i+3]))
			return aa[::-1]
	
	def counts(self, seq, strand):
		aa = self.sequence(seq, strand)
		return Counter(aa)

	def frequencies(self, seq, strand):
		counts = self.counts(seq, strand)
		total = sum(counts.values())
		for aa in counts:
			counts[aa] = counts[aa] / total
		return counts

	def dimers(self, seq, strand):
		peptides = self.sequence(seq, strand)
		counts = { (a,b):0 for a in self.amino_acids + ['X'] for b in self.amino_acids + ['X'] }
		for i in range(len(peptides)-1):
			counts[ ( peptides[i] , peptides[i+1] ) ] += 1
		t = sum(counts.values()) if sum(counts.values()) else 1
		return [ counts[(a,b)]/t for a in self.amino_acids for b in self.amino_acids ]

translate = Translate()


class Locus(dict):
	def __init__(self, locus, dna):
		self.locus = locus
		self.dna = dna
		self.locations = cd.Locations(self.dna)
		
		seq = self.dna + rev_comp(self.dna)
		length = len(seq)
		self.p = dict()
		self.p['a'] = seq.count('a') / length
		self.p['c'] = seq.count('c') / length
		self.p['g'] = seq.count('g') / length
		self.p['t'] = seq.count('t') / length

	def seq(self, left, right):
		return self.dna[left-1 : right]

	def length(self):
		return len(self.dna)

	def pcodon(self, codon):
		codon = codon.lower()
		return self.p[codon[0]] * self.p[codon[1]] * self.p[codon[2]]

	def pstart(self):
		return self.pcodon('atg') + self.pcodon('gtg') + self.pcodon('ttg')

	def pstop(self):
		return self.pcodon('taa') + self.pcodon('tag') + self.pcodon('tga')
		
	def check(self):	
		# set dna for features and check integrity
		last = None
		for feature in sorted(self):
			# this just mkes sure the feature locations are in the same frame
			for i in feature.base_locations():
				feature.dna += self.dna[ i-1 : i ]
			#if feature.type == 'CDS':
			#	if len(feature.dna) % 3 and not feature.partial and 'transl_except' not in feature.tags:
			#		raise ValueError("Out of frame: %s" % feature)
			if last is not None and (feature.type == 'CDS') and (feature.frame() == last.frame()):
				seq = self.seq(last.right()-29 , feature.left()+30)
				last.tags['seq'] = seq
				if not has_stop(seq):
					del self[last]
					del self[feature]
					last.pairs = ((last.left() , feature.right()),)
					last.tags['merged'] = 'true'
					self[last] = True
			else:
				last = feature
		last = None
		for feature in sorted(self):
			if last is not None and (last.type == 'CDS') and (feature.type == 'CDS') and (feature.strand == last.strand):
				if (feature.strand > 0) and (abs(last.stop_distance()) > 100):
					del self[last]
					del self[feature]
					last.pairs = last.pairs + feature.pairs
					last.tags['joined'] = 'true'
					self[last] = True
					continue
			last = feature


	def features(self, include=None, exclude=None):
		for feature in self:
			if not include or feature.type in include:
				yield feature

	def add_feature(self, key, strand, pairs):
		"""Add a feature to the factory."""
		feature = Feature(key, strand, pairs, self)
		feature.locus = self.locus
		if feature not in self:
			self[feature] = True

	def gene_coverage(self):
		''' This calculates the protein coding gene coverage, which should be around 1 '''
		cbases = tbases = 0	
		for locus in self.values():
			dna = [False] * len(locus.dna)
			seen = dict()
			for feature in locus.features(include=['CDS']):
				for i in feature.codon_locations():
					dna[i-1] = True
			cbases += sum(dna)
			tbases += len(dna)
		return 3 * cbases / tbases

	def write(self, outfile):
		outfile.write('LOCUS       ')
		outfile.write(self.locus)
		outfile.write(str(len(self.dna)).rjust(10))
		outfile.write(' bp    DNA             UNK')
		outfile.write('\n')
		outfile.write('DEFINITION  ' + self.locus + '\n')
		outfile.write('FEATURES             Location/Qualifiers\n')
		outfile.write('     source          1..')
		outfile.write(str(len(self.dna)))
		outfile.write('\n')

		for feature in sorted(self):
			feature.write(outfile)
		outfile.write('//')
		outfile.write('\n')

			
class Feature(Locus):
	def __init__(self, type_, strand, pairs, locus):
		super().__init__(locus.locus, locus.dna)
		self.type = type_
		self.strand = strand
		# tuplize the pairs
		self.pairs = tuple([tuple(pair) for pair in pairs])
		self.tags = dict()
		self.dna = ''
		self.partial = False

	def frame(self):
		if self.type != 'CDS':
			return None
		else:
			if self.left()%3 != (self.right()-2)%3:
				print(self)
			assert self.left()%3 == (self.right()-2)%3
			return self.left()%3 * self.strand

	def hypothetical(self):
		function = self.tags['product'] if 'product' in self.tags else ''
		if 'hypot'  in function or \
		   'etical' in function or \
		   'unchar' in function or \
		   ('orf' in function and 'orfb' not in function):
			return True
		else:
			return False

	def left(self):
		return int(self.pairs[0][0])
	
	def right(self):
		return int(self.pairs[-1][-1])

	def nearest_start(self):
		if self.strand > 0:
			return self.locations.nearest_start(self.left(),'+')
		else:
			return self.locations.nearest_start(self.right()-2,'-')

	def nearest_stop(self):
		if self.strand < 0:
			return self.locations.nearest_stop(self.left(),'-')
		else:
			return self.locations.nearest_stop(self.right()-2,'+')

	def start_distance(self):
		if self.strand > 0:
			return self.left() - self.nearest_start()
		else:
			return self.nearest_start() - (self.right()-2)

	def stop_distance(self):
		if self.strand > 0:
			return self.nearest_stop() - (self.right()-2)
		else:
			return self.left() - self.nearest_stop()

	def __str__(self):
		"""Compute the string representation of the feature."""
		return "%s\t%s\t%s\t%s" % (
				repr(self.locus),
				repr(self.type),
				repr(self.pairs),
				repr(self.tags))

	def __repr__(self):
		"""Compute the string representation of the feature."""
		return "%s(%s, %s, %s, %s)" % (
				self.__class__.__name__,
				repr(self.locus),
				repr(self.type),
				repr(self.pairs),
				repr(self.tags))
	def __hash__(self):
		return hash(self.pairs)
	#def __eq__(self, other):
	#	return self.pairs == other.pairs()

	def __lt__(self, other):
		return self.left() < other.left()

	def base_locations(self, full=False):
		if full and self.partial == 'left': 
			for i in range(-((3 - len(self.dna) % 3) % 3), 0, 1):
				yield i+1
		for left,right in self.pairs:
			#left,right = map(int, [ item.replace('<','').replace('>','') for item in pair ] )
			for i in range(left,right+1):
				yield i

	def codon_locations(self):
		assert self.type == 'CDS'
		for triplet in grouper(self.base_locations(full=True), 3):
			if triplet[0] >= 1:
				yield triplet

	def translation(self):
		global translate
		aa = []
		codon = ''
		first = 0 if not self.partial else len(self.dna) % 3
		for i in range(first, len(self.dna), 3):
			codon = self.dna[ i : i+3 ]
			if self.strand > 0:
				aa.append(translate.codon(codon))
			else:
				aa.append(translate.codon(rev_comp(codon)))
		if self.strand < 0:
			aa = aa[::-1]
		if aa[-1] in '#*+':
			aa.pop()
		#aa[0] = 'M'
		return "".join(aa)

	def integrity_check(self):
		seq2 = self.translation()
		if 'translation' not in self.tags:
			return 1 - ( seq2.count('#') + seq2.count('*') + seq2.count('+') ) / len(seq2)
		else:
			seq1 = self.tags['translation']
			seq1,seq2 = mask(seq1, seq2)
			seq1,seq2 = (seq1[1:], seq2[1:])
			return max(
					fuzz.ratio(seq1, seq2),
					fuzz.ratio(seq1, seq2.replace('*', 'W'))
					) / 100


	def write(self, outfile):

		outfile.write('     ')
		outfile.write( self.type.ljust(16) )
		if not self.strand > 0:
			outfile.write('complement(')
		# the pairs
		if len(self.pairs) > 1:
			outfile.write('join(')
		pairs = []
		for left, right in self.pairs:
			left = max(1,left)
			pair = str(left) + '..' + str(right)
			pairs.append(pair)
		outfile.write(','.join(pairs))
		if len(self.pairs) > 1:
			outfile.write(')')
		# the pairs
		if not self.strand > 0:
			outfile.write(')')
		outfile.write('\n')
		outfile.write('                     /colour=100 100 100')
		outfile.write('\n')
		for key,value in self.tags.items():
			outfile.write('                     /')
			outfile.write(key)
			outfile.write('=')
			outfile.write(value)
			outfile.write('\n')
		
		if self.type == 'CDS':
			outfile.write('                     /nearest_start=')
			#outfile.write( str(self.nearest_start) )
			outfile.write(str( self.start_distance() ))
			outfile.write('\n')
			outfile.write('                     /nearest_stop=')
			#outfile.write( str(self.nearest_stop) )
			outfile.write(str( self.stop_distance() ))
			outfile.write('\n')

			outfile.write('                     /nearest_start_prob=')
			outfile.write(str( (1-self.pstart()) ** abs(2*self.start_distance()/3)  ))
			outfile.write('\n')
			outfile.write('                     /nearest_stop_prob=')
			outfile.write(str( (1-self.pstop()) ** abs(2*self.stop_distance()/3)  ))
			outfile.write('\n')

		

	

if __name__ == "__main__":
	usage = '%s [-opt1, [-opt2, ...]] infile' % __file__
	parser = argparse.ArgumentParser(description='', formatter_class=RawTextHelpFormatter, usage=usage)
	parser.add_argument('infile', type=is_valid_file, help='input file in genbank format')
	parser.add_argument('-o', '--outfile', action="store", default=sys.stdout, type=argparse.FileType('w'), help='where to write output [stdout]')
	parser.add_argument('-f', '--format', help='Output the features in the specified format', type=str, default='tabular', choices=['tabular','genbank','fasta', 'fna','faa', 'coverage'])
	args = parser.parse_args()

	genbank = GenbankFile(args.infile)

	if args.format == 'tabular':
		for feature in genbank.features(include=['CDS']):
			args.outfile.write(str(feature))
			args.outfile.write("\n")
	elif args.format == 'coverage':
		args.outfile.write(str(genbank.gene_coverage()))
		args.outfile.write("\n")
	elif args.format == 'faa':
		for feature in genbank.features(include=['CDS']):
			args.outfile.write(">")
			args.outfile.write(feature.locus)
			args.outfile.write("[")
			args.outfile.write(feature.location.split()[1])
			args.outfile.write("]")
			args.outfile.write("\n")
			args.outfile.write(feature.translation())
			args.outfile.write("\n")

