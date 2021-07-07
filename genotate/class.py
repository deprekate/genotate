import sys


f = open(sys.argv[1])
for i,row in enumerate(f):
	#if not i%2:
		#print(1+i//2, p[i], p[i+1])
	#print(1+i//2, p[i])
	#continue	
	if int(row) == 1:
		if i%2:
			print('     CDS             complement(', ((i-1)//2)+1, '..', ((i-1)//2)+3, ')', sep='')
		else:
			print('     CDS             ', (i//2)+1 , '..', (i//2)+3, sep='')
		print('                     /colour=100 100 100')
print("//")
