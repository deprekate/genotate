import os
import sys
import re
import argparse
from argparse import RawTextHelpFormatter
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from genbank.file import File
from genotate.make_train import get_windows, parse_genbank
from genotate.make_model import create_model_blend

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
	tf.config.experimental.set_memory_growth(physical_devices[0], True)

def is_valid_file(x):
	if not os.path.exists(x):
		raise argparse.ArgumentTypeError("{0} does not exist".format(x))
	return x


def pack(features):
	#return tf.stack(list(features.values()), axis=-1)
	labels,datas,windows = tf.split(features, [1,3,1], axis=-1)
	labels = tf.strings.to_number(labels, out_type=tf.dtypes.int32)
	labels = tf.one_hot(labels, depth=3)
	labels = tf.reshape(labels, [-1])

	#return (features, tf.stack([features2, features3, features4], axis=-1)), tf.one_hot(labels, depth=3)
	#return (tf.expand_dims(windows, axis=-1), tf.expand_dims(datas, axis=-1)), tf.one_hot(labels, depth=3)
	#windows = tf.stack([windows], axis=-1)
	#windows = tf.expand_dims([windows], axis=-1)
	#datas = tf.stack([datas], axis=-1)
	#datas = tf.unstack(datas, axis=0)
	#datas = tf.expand_dims([datas], axis=0)
	print(windows)
	print(datas)
	print(labels)
	return (windows, datas) , labels

def parse_fn(filename):
	return parse_genbank(filename.decode())

if __name__ == '__main__':
	usage = '%s [-opt1, [-opt2, ...]] directory' % __file__
	parser = argparse.ArgumentParser(description='', formatter_class=RawTextHelpFormatter, usage=usage)
	parser.add_argument('directory', type=is_valid_file, help='input directory')
	parser.add_argument('-o', '--outfile', action="store", default=sys.stdout, type=argparse.FileType('w'), help='where to write output [stdout]')
	parser.add_argument('-k', '--kfold', action="store", default=0, type=int, help='which kfold')
	parser.add_argument('-t', '--trim', action="store", default=0, type=int, help='how many bases to trim off window ends')
	parser.add_argument('-r', '--reg', action="store_true", help='use kernel regularizer')
	args = parser.parse_args()

	filenames = ["ten/" + str(i) + '.gbk' for i in range(10)]

	dataset = tf.data.Dataset.from_tensor_slices(filenames)

	dataset = dataset.interleave(
								lambda x: tf.data.Dataset.from_generator(
									parse_fn,
									args=(x,),
									output_signature=(
										tf.TensorSpec(shape=(5,),dtype=tf.string)
									)
								),
								num_parallel_calls=10, #tf.data.AUTOTUNE,
								cycle_length=10,
								block_length=100
								)
	
	print(dataset)
	tdata = dataset.map(pack)
	tdata = tdata.batch(10)
	#for feature in tdata.take(1):
	#	print( feature )
	#exit()
	print(tdata)
	
	model = create_model_blend(args)
	with tf.device('/device:GPU:0'):
		model.fit(
				  tdata,
				  epochs          = 10,
				  verbose         = 1,
		)
