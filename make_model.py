import os

# TensorFlow and tf.keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

def create_model(opt):
	'''
	This creates and returns a new model
	'''
	model = tf.keras.Sequential([
					tf.keras.layers.Dense(21, input_shape=(21,)),
					#tf.keras.layers.Dense(41, input_shape=(41,)),
					#tf.keras.layers.Dropout(0.05),
					tf.keras.layers.Dense(33, activation='relu'),
					tf.keras.layers.Dense(3, activation='softmax')
	])
	model.compile(optimizer = opt,
				  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
				  metrics=['accuracy']
				  )
	return model
