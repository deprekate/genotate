import os
import sys
import argparse
from argparse import RawTextHelpFormatter

import tensorflow as tf
import numpy as np

def is_valid_file(x):
	#if not os.path.exists(x):
	#	raise argparse.ArgumentTypeError("{0} does not exist".format(x))
	return x

usage = '%s [-opt1, [-opt2, ...]] infile' % __file__
parser = argparse.ArgumentParser(description='', formatter_class=RawTextHelpFormatter, usage=usage)
parser.add_argument('checkpoint', type=is_valid_file, help='input file')
parser.add_argument('-o', '--outfile', action="store", default=sys.stdout, type=argparse.FileType('w'), help='where to write output [stdout]')
args = parser.parse_args()

np.set_printoptions(threshold=np.inf)

print("from numpy import array")
print("from numpy import float32")
ckpt_reader = tf.train.load_checkpoint(args.checkpoint)
for i in range(10):
	name = "layer_with_weights-%i/kernel/.ATTRIBUTES/VARIABLE_VALUE" % i
	if name in ckpt_reader.get_variable_to_shape_map():
		print("layer%s_weights =" % i,  repr(ckpt_reader.get_tensor(name)) )
	name = "layer_with_weights-%i/bias/.ATTRIBUTES/VARIABLE_VALUE" % i
	if name in ckpt_reader.get_variable_to_shape_map():
		print("layer%s_bias ="    % i,  repr(ckpt_reader.get_tensor(name)) )
