import os
import sys
from decimal import Decimal

import numpy as np
import tensorflow as tf
from types import ModuleType, FunctionType
from gc import get_referents

# Custom objects know their class.
# Function objects seem to know way too much, including modules.
# Exclude modules as well.
BLACKLIST = type, ModuleType, FunctionType
def getsize(obj):
    """sum size of object & members."""
    if isinstance(obj, BLACKLIST):
        raise TypeError('getsize() does not take argument of type: '+ str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size




def ave(a):
	return round(sum(a)/len(a), 2)

def rev_comp(seq):
	seq_dict = {'a':'t','t':'a','g':'c','c':'g',
		    'n':'n',
		    'r':'y','y':'r','s':'s','w':'w','k':'m','m':'k',
		    'b':'v','v':'b','d':'h','h':'d'}
	return "".join([seq_dict[base] for base in reversed(seq)])

def write_codons(p):
	Y = np.argmax(p,axis=-1)
	#Y = smooth(Y)
	#Y = cutoff(Y)
	for i,row in enumerate(Y[:-4]):
		#if not i%2:
		#	print(1+i//2, p[i], p[i+1], sep='\t')
		#print(1+i//2, p[i], sep='\t')
		#continue	
		if row == 1:
			if i%2:
				print('     CDS             complement(', ((i-1)//2)+1, '..', ((i-1)//2)+3, ')', sep='')
			else:
				print('     CDS             ', (i//2)+1 , '..', (i//2)+3, sep='')
			print('                     /colour=100 100 100')
	print("//")


def plot_strands(p):
	#p = smo(p, 30)
	#p = smoo(p)
	forward = np.array([ p[0::6,:] , p[2::6,:] , p[4::6,:] ])
	reverse = np.array([ p[1::6,:] , p[3::6,:] , p[5::6,:] ])

	strand_wise = np.array([ 
							reverse[:,:,1].sum(axis=0).clip(0,1) , 
							np.divide( reverse[:,:,2] + forward[:,:,2], 6).sum(axis=0).clip(0,1) , 
							forward[:,:,1].sum(axis=0).clip(0,1) 
							]).T
	# detection
	#algo = rpt.Pelt(model="rbf").fit(signal)
	result = rpt.KernelCPD(kernel="linear", min_size=33).fit(strand_wise).predict(pen=15)
	result = [x*3 for x in result]
	
	print("# BASE VAL1  VAL2 VAL3  VAL4")
	print("# colour 255:0:0 0:0:255 0:0:0 128:128:128 0:0:0")
	for i in range(len(strand_wise)):
		c = 1 if 3*i in result or 3*i+1 in result or 3*i+2 in result or 3*i+3 in result or 3*i+4 in result or 3*i+5 in result or 3*i+6 in result or 3*i+7 in result or 3*i+8 in result or 3*i+9 in result  else 0
		print(3*i+1, strand_wise[i,0], strand_wise[i,1], strand_wise[i,2], c, sep='\t')
		print(3*i+2, strand_wise[i,0], strand_wise[i,1], strand_wise[i,2], c, sep='\t')
		print(3*i+3, strand_wise[i,0], strand_wise[i,1], strand_wise[i,2], c, sep='\t')
	exit()

def plot_frames(p):
	print("# BASE VAL1  VAL2 VAL3  VAL4 VAL5 VAL6 VAL7")
	print("# colour 255:0:0 0:0:255 0:0:0 255:0:255 0:128:128 128:128:128 255:195:0")
	for i in range(0,len(p), 6):
		val = []
		ig = []
		for j in range(6):
			val.append(p[i+j,1])
			ig.append(p[i+j, 2])
		v = [None] * 7
		v[0] = val[0]
		v[1] = val[2]
		v[2] = val[4]
		v[3] = val[1]
		v[4] = val[3]
		v[5] = val[5]
		v[6] = max(ig)
		print(i // 2 + 1, end='\t')
		print('\t'.join(map(str,v)))
		print(i // 2 + 2, end='\t')
		print('\t'.join(map(str,v)))
		print(i // 2 + 3, end='\t')
		print('\t'.join(map(str,v)))
	exit()

def to_dna(s):
	to_base = {0:'n',1:'a',2:'c',3:'t',4:'g'}
	dna = ''
	for num in s:
		dna += to_base[num]	
	return dna

def skew(seq, nucs):
	windowsize = stepsize = 32 #int(len(self.sequence) / 1000)
	(nuc_1,nuc_2) = nucs
	#return [0] * len(seq)
	cumulative = 0
	cm_list = []
	i = int(windowsize / 2)
	for each in range(len(seq) // stepsize):
		if i < len(seq):
			a = seq[i - windowsize//2:i + windowsize // 2].count(nuc_1)
			b = seq[i - windowsize//2:i + windowsize // 2].count(nuc_2)
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


def parse_locus(locus):
		#def iter_genbank(infile):
		#for locus in File(infile.decode()):
		# label the positions
		positions = dict()
		for feature in locus.features(include=['CDS']):
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
		#for k,v in positions.items():
		#	print(k,v)
		#exit()
		dna = locus.seq()
		#at_skew = np.array(skew(dna, 'at'))
		#gc_skew = np.array(skew(dna, 'gc'))
		forward = np.zeros(48+len(dna)+50)
		reverse = np.zeros(48+len(dna)+50)
		for i,base in enumerate(dna):
			#if base in 'acgt':
			forward[i+48] = ((ord(base) >> 1) & 3) + 1
			reverse[i+48] = ((forward[i+48] - 3) % 4) + 1
		a = np.zeros([6, 100], dtype=int)
		#a[:,100] = locus.gc_content() 
		for n in range(0, len(dna)-2, 3):
			#pos = n//100
			i = n + 0
			a[0,0] = positions.get( i, 2)
			a[1,0] = positions.get(-i, 2)
			a[0,1:100] = forward[i : i+99 ]
			a[1,1:100] = reverse[i : i+99 ][::-1]
			i = n + 1
			a[2,0] = positions.get( i, 2)
			a[3,0] = positions.get(-i, 2)
			a[2,1:100] = forward[i : i+99 ]
			a[3,1:100] = reverse[i : i+99 ][::-1]
			i = n + 2
			a[4,0] = positions.get( i, 2)
			a[5,0] = positions.get(-i, 2)
			a[4,1:100] = forward[i : i+99 ]
			a[5,1:100] = reverse[i : i+99 ][::-1]
			#a[2*f  ,101] =  at_skew[pos]
			#a[2*f+1,101] = -at_skew[pos]
			#a[2*f  ,102] =  gc_skew[pos]
			#a[2*f+1,102] = -gc_skew[pos]
			yield a
