from math import log
from math import exp

def pgp(A):
	x = 0
	for a in A:
		x += log( a / (1-a)) if a else 0
	return 1 / ( 1 + exp( -x / len(A)))


print(pgp([0.9,0,0.3]))
