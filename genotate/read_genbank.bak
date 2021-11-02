import sys
import re


class Feature:
	def __init__(self, line):
		self.type = line.split()[0]
		self.direction = -1 if 'complement' in line else 1
		pairs = [pair.split('..') for pair in re.findall(r"<*\d+\.\.>*\d+", line)]
		# this is for weird malformed features
		if ',1)' in line:
			pairs.append(['1','1'])
		# tuplize the pairs
		self.pairs = tuple([tuple(pair) for pair in pairs])
		self.tags = dict()
		self.line = line
		#if remainder and ">" not in pair[1] and 'trna' not in line:
		#	raise ValueError("Out of frame: ( %s , %s )" % tuple(pair))

	def hypothetical(self):
		function = self.tags['product'] if 'product' in self.tags else ''
		if 'hypot'  in function or \
		   'etical' in function or \
		   'unchar' in function or \
		   ('orf' in function and 'orfb' not in function):
			return True
		else:
			return False

	def __str__(self):
		"""Compute the string representation of the feature."""
		return "%s\t%s\t%s" % (
				repr(self.type),
				repr(self.pairs),
				repr(self.tags)
				)

	def __repr__(self):
		"""Compute the string representation of the edge."""
		return "%s(%s, %s, %s)" % (
				self.__class__.__name__,
				repr(self.type),
				repr(self.pairs),
				repr(self.tags))

class GenbankFeatures(dict):
	def __init__(self, filename, current=None):
		self.current = current
		self.dna = False
		in_features = False
		with open(filename) as fp:
			for line in fp:
				if line.startswith('ORIGIN'):
					in_features = False
					self.dna = ''
				elif line.startswith('FEATURES'):
					in_features = True
				elif in_features:
					line = line.rstrip()
					if not line.startswith(' ' * 21):
						if line.rstrip().endswith(','):
							line += next(fp).strip()
						self.add_feature(line)
					else:
						while line.count('"') == 1:
							line += next(fp).strip()
						tag,_,value = line[22:].partition('=')
						self.current.tags[tag] = value.replace('"', '')
				elif self.dna != False:
					self.dna += line[10:].rstrip().replace(' ','').lower()

	def features(self, include=None, exclude=None):
		for  feature in self:
			if feature.type in include:
				yield feature

	def add_feature(self, line):
		"""Add a feature to the factory."""
		feature = Feature(line)
		if feature not in self:
			self[feature] = True
			self.current = feature

if __name__ == "__main__":
	genbank = GenbankFeatures(sys.argv[1])
	for feature in genbank.features(include=['CDS']):
		print(feature)


