import os
import sys
import argparse
from argparse import RawTextHelpFormatter
from statistics import mode

import make_model as mm

# TensorFlow and tf.keras
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

#sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

# Helper libraries
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd

def is_valid_file(x):
	if not os.path.exists(x):
		raise argparse.ArgumentTypeError("{0} does not exist".format(x))
	return x

def pack(features, label):
  return tf.stack(list(features.values()), axis=-1), label


if __name__ == '__main__':
	usage = '%s [-opt1, [-opt2, ...]] directory' % __file__
	parser = argparse.ArgumentParser(description='', formatter_class=RawTextHelpFormatter, usage=usage)
	parser.add_argument('directory', type=is_valid_file, help='input directory')
	parser.add_argument('-o', '--outfile', action="store", default=sys.stdout, type=argparse.FileType('w'), help='where to write the output [stdout]')
	args = parser.parse_args()

	letters = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
	#colnames = ['TYPE','GC'] + letters
	#colnames = ['TYPE','GC'] + [letter for pair in zip([l+'1' for l in letters], [l+'2' for l in letters]) for letter in pair]
	colnames =  ['TYPE', 'GC'] + [letter+f for letter in letters for f in ['+0','-0','+1','-1','+2','-2']]
	tfiles = tf.data.experimental.make_csv_dataset(
		file_pattern        = args.directory + "/A*.tsv",
		field_delim         = '\t',
		header              = False,
		column_names        = colnames,
		select_columns      = colnames,
		column_defaults     = [tf.int32] + [tf.float32] * (len(colnames)-1),
		batch_size          = 100,
		num_epochs          = 1,
		shuffle_buffer_size = 10000,
		num_parallel_reads  = 10,
		sloppy              = True,
		label_name          = colnames[0]
		#label_name          = 0 # this is old version
		)
	pdata = tfiles.map(pack)
	#for feature in tfiles.take(1):
	#	print( feature )
	with tf.device('/device:CPU:0'):
		cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=args.directory + '.ckpt', save_weights_only=True, verbose=1)
		model = mm.create_model3('adam')
		model.fit(pdata, epochs=10, callbacks=[cp_callback])



