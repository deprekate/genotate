import os
import sys
from decimal import Decimal


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
