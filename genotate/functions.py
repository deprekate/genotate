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


