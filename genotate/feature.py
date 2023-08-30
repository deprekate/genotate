from genbank.feature import Feature
import textwrap


class Feature(Feature):

	def more(self):
		return "mooore"

	def end(self):
		if self.strand > 0:
			return self.right()
		else:
			return self.left()
	
	def start_codon(self):
		return self.seq()[:3]

	def stop_codon(self):
		return self.seq()[-3:]

	def nearest_start(self):
		if self.strand > 0:
			return self.locus.locations.nearest_start(self.left(),'+')
		else:
			return self.locus.locations.nearest_start(self.right(),'-')

	def nearest_stop(self):
		if self.strand < 0:
			return self.locus.locations.nearest_stop(self.left(),'-')
		else:
			return self.locus.locations.nearest_stop(self.right(),'+')

	def start_distance(self):
		if self.strand > 0:
			return self.left() - self.nearest_start()
		else:
			return self.nearest_start() - (self.right())

	def stop_distance(self):
		if self.strand > 0:
			return self.nearest_stop() - (self.right())
		else:
			return self.left() - self.nearest_stop()

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

	def set_left(self, left):
		if left:
			pairs = [list(tup) for tup in self.pairs]
			pairs[0][0] = str(left)
			self.pairs = tuple([tuple(lis) for lis in pairs])

	def set_right(self, right):
		if right:
			pairs = [list(tup) for tup in self.pairs]
			pairs[-1][-1] = str(right)
			self.pairs = tuple([tuple(lis) for lis in pairs])


