import os
import sys
from itertools import zip_longest, chain, tee, islice
from termcolor import colored
from math import log10, exp, sqrt
import pickle
import pkgutil
import pkg_resources
from math import log

import score_rbs

from genbank.locus import Locus
from genotate.feature import Feature



def rint(s):
	return int(s.replace('<','').replace('>',''))


def rround(item, n=4):
    try:
        return round(item, n)
    except:
        try:
            return item.decode()
        except:
            return item

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
	#def __init__(self,parent, *args, **kwargs):
		#self = dict(parent)
		#self.__class__ = type(parent.__class__.__name__,(self.__class__, parent.__class__),{})
		#self.__dict__ = parent.__dict__
		#self.update(dict(parent))
		#for key,value in parent.items():
		#	self[key] = value

	def init(self, args):
		#self._rbs = score_rbs.ScoreXlationInit()
		self.stops = ['taa','tga','tag']
		pass

	def score_rbs(self, rbs):
		return self._rbs.score_init_rbs(rbs,20)[0]

	def merge(self):	
		# set dna for features and check integrity
		_last = _curr = None
		for _, _, _next in previous_and_next(sorted(self)):
			# this just mkes sure the feature locations are in the same frame
			#for i in _curr.base_locations():
			#	_curr.dna += self.dna[ i-1 : i ]
			if _last is None or (_last.type != 'CDS') or (_curr.type != 'CDS'):
				pass
			elif _curr.strand != _last.strand:
				pass
			elif _curr.frame('left') == _last.frame('right'):
				# this merges adjacent frames
				seq = self.seq(_last.right()-30 , _curr.left()+32)
				if not has_stop(seq, _last.strand):
					del self[_last]
					del self[_curr]
					_last.tags['note'] = ['"merged:' + str(_last.pairs) + str(_curr.pairs) + '"' ]
					#_last.pairs = ( (_last.left() , _curr.right()), )
					_last.pairs = ( (_last.pairs[0][0] , _curr.pairs[-1][-1]), )
					#_last.tags['seq'] = [seq]
					#_last.tags['other'] = [_last.pairs]
					self[_last] = True
					_curr = _next
					continue
			elif _next is None:
				pass
			elif _last.frame('right') == _next.frame('left'):
				# this merges frames broken by an embedded gene
				seq = self.seq(_last.right()-30 , _next.left()+32)
				if not has_stop(seq, _last.strand):
					del self[_last]
					del self[_next]
					_last.tags['note'] = '"merged: ' + str(_last.pairs) + ' and ' + str(_curr.pairs) + '"'
					_last.pairs = ((_last.left() , _next.right()),)
					#_last.tags['seq'] = [seq]
					#_last.tags['pstop'] = (1-self.pstop()) ** (len(seq)/3)
					#_last.tags['pstopl'] = (1-self.pstop(rev_comp(seq))) ** (len(seq)/3)
					#_last.tags['merged'] = 'true'
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
