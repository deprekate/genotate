import os
import sys
import argparse
from argparse import RawTextHelpFormatter
from statistics import mode

import make_model as mm

# TensorFlow and tf.keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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

class Metrics(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self._data = []

    def on_epoch_end(self, batch, logs={}):
        x_test, y_test = self.validation_data[0], self.validation_data[1]
        y_predict = np.asarray(model.predict(x_test))

        true = np.argmax(y_test, axis=1)
        pred = np.argmax(y_predict, axis=1)

        cm = confusion_matrix(true, pred)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        self._data.append({
            'classLevelaccuracy':cm.diagonal() ,
        })
        return

    def get_data(self):
        return self._data

if __name__ == '__main__':
	usage = '%s [-opt1, [-opt2, ...]] directory' % __file__
	parser = argparse.ArgumentParser(description='', formatter_class=RawTextHelpFormatter, usage=usage)
	parser.add_argument('directory', type=is_valid_file, help='input directory')
	parser.add_argument('-o', '--outfile', action="store", default=sys.stdout, type=argparse.FileType('w'), help='where to write output [stdout]')
	args = parser.parse_args()

	letters = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
	#colnames = ['TYPE','GC'] + letters
	#colnames = ['TYPE','GC'] + [letter for pair in zip([l+'1' for l in letters], [l+'2' for l in letters]) for letter in pair]
	colnames =  ['TYPE', 'GC', 'DYN'] + [letter+f for letter in letters for f in ['+0','-0','+1','-1','+2','-2']]
	selnames = ['TYPE'] + colnames[3:]
	tfiles = tf.data.experimental.make_csv_dataset(
		file_pattern        = args.directory + "/AB0*.tsv",
		field_delim         = '\t',
		header              = False,
		column_names        = colnames,
		select_columns      = selnames,
		column_defaults     = [tf.int32] + [tf.float32] * (len(selnames)-1),
		batch_size          = 100,
		num_epochs          = 1,
		shuffle_buffer_size = 1000,
		num_parallel_reads  = 10,
		sloppy              = True,
		label_name          = colnames[0]
		#label_name          = 0 # this is old version
		)
	pdata = tfiles.map(pack)
	#for feature in tfiles.take(1):
	#	print( feature )
	#metrics = Metrics()
	#class_weight = {0:0.5, 1:1, 2:5}
	
	model = mm.create_model3(len(selnames)-1)

	class_weight = {0:1, 1:1}
	with tf.device('/device:CPU:0'):
		cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=args.directory + '.ckpt', save_weights_only=True, verbose=1)
		model.fit(pdata,
				  epochs=9,
				  class_weight=class_weight,
				  callbacks=[cp_callback]
				  )



