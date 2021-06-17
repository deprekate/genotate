import os

# TensorFlow and tf.keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

def create_model(i):
	'''
	This creates and returns a new model
	'''
	model = tf.keras.Sequential([
					tf.keras.layers.Dense(i, input_shape=(i,)),
					tf.keras.layers.Dense(i * 10, activation='relu'),
					tf.keras.layers.Dense(i * 3, activation='relu'),
					tf.keras.layers.Dense(3, activation='softmax')
	])
	model.compile(optimizer = 'adam',
				  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
				  metrics=['accuracy']
				  )
	return model


def create_model3(i):
	'''
	This creates and returns a new model
	'''
	model = tf.keras.Sequential([
					tf.keras.layers.Dense(i, input_shape=(i,)),
					#tf.keras.layers.Dropout(0.05),
					tf.keras.layers.Dense(1666, activation='relu'),
					tf.keras.layers.Dense(166, activation='relu'),
					tf.keras.layers.Dense(3, activation='softmax')
	])
	model.compile(optimizer = 'adam',
				  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
				  metrics=['accuracy']
				  )
	return model


def create_model4(i):
	'''
	This creates and returns a new model
	'''
	model = tf.keras.Sequential([
					tf.keras.layers.Dense(i, input_shape=(i,)),
					tf.keras.layers.Dense(i*10, activation='relu'),
					tf.keras.layers.Dense(i*3, activation='relu'),
					tf.keras.layers.Dense(1, activation='sigmoid')
	])
	model.compile(optimizer = 'adam',
				  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
				  metrics=['accuracy']
				  )
	return model
