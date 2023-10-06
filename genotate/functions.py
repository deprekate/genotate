import os
import sys
from decimal import Decimal

import numpy as np
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


def plot_strands(args, p):
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

def plot_frames(args, p):
	args.outfile.print("# BASE VAL1  VAL2 VAL3  VAL4 VAL5 VAL6 VAL7\n")
	args.outfile.print("# colour 255:0:0 0:0:255 0:0:0 255:0:255 0:128:128 128:128:128 255:195:0\n")
	for i in range(0,len(p) // 6 * 6, 6):
		val = []
		ig = []
		for j in range(6):
			val.append('%f' % p[i+j,1])
			ig.append('%f' % p[i+j, 2])
		v = [None] * 7
		v[0] = val[0]
		v[1] = val[2]
		v[2] = val[4]
		v[3] = val[1]
		v[4] = val[3]
		v[5] = val[5]
		v[6] = max(ig)
		args.outfile.print(i // 2 + 1)
		args.outfile.print('\t')
		args.outfile.print('\t'.join(map(str,v)))
		args.outfile.print('\n')
		args.outfile.print(i // 2 + 2 )
		args.outfile.print('\t')
		args.outfile.print('\t'.join(map(str,v)))
		args.outfile.print('\n')
		args.outfile.print(i // 2 + 3)
		args.outfile.print('\t')
		args.outfile.print('\t'.join(map(str,v)))
		args.outfile.print('\n')

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
		w = 87
		n = 20000
		X = np.zeros([2*n ,w],dtype=np.uint8)
		length = locus.length()
		Y = np.zeros([length*2+7,3],dtype=np.uint8)
		Y[:,2] = 1
		# label the positions
		for feature in locus.features(include=['CDS']):
			s = (feature.strand >> 1) * -1
			locations = feature.codon_locations()
			if feature.partial() == 'left': next(locations)
			for i,*_ in locations:
				i = 2 * i
				# coding
				Y[i+s  ,1] = 1
				# noncoding
				Y[i:i+6,0] = 1
				Y[i:i+6,2] = 0
		Y[Y[:,1]==1,0] = 0
		forward = np.zeros((w//2-1)+length+(w//2+1),dtype=np.uint8)
		reverse = np.zeros((w//2-1)+length+(w//2+1),dtype=np.uint8)
		for i,base in enumerate(locus.dna):
			i += w//2-1
			# I can prob delete this line if need 4 speed
			if base in 'AaCcGgTt':
				forward[i] = ((ord(base) >> 1) & 3) + 1
				reverse[i] = ((forward[i] - 3) % 4) + 1
		# this splits the BIG numpy array into n sized chunks to limit ram usage
		for i in range( length // n):
			i *= n
			X[0::2,] = np.lib.stride_tricks.sliding_window_view(forward[ i : i+n+w-1],w)
			X[1::2,] = np.lib.stride_tricks.sliding_window_view(reverse[ i : i+n+w-1],w)[:,::-1]
			yield X,Y[ i*2:i*2+n*2 , :]
		i = length // n * n
		r = length % n
		if r:
			X[0:2*r:2,] = np.lib.stride_tricks.sliding_window_view(forward[ i : i+r+w-1],w)
			X[1:2*r:2,] = np.lib.stride_tricks.sliding_window_view(reverse[ i : i+r+w-1],w)[:,::-1]
			yield X[ : 2*r , : ] , Y[ i*2: i*2+2*r , :]

