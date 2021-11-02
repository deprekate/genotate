import os

# TensorFlow and tf.keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


def create_model(i):
	model = tf.keras.Sequential([
					tf.keras.layers.Dense(i, input_shape=(i,)),
					tf.keras.layers.Dense(i * 10, activation='relu'),
					tf.keras.layers.Dense(i * 3, activation='relu'),
					#tf.keras.layers.Dense(i * 2, activation='relu'),
					#tf.keras.layers.Dense(i * 1, activation='relu'),
					tf.keras.layers.Dense(3, activation='softmax')
	])
	opt = tf.keras.optimizers.Adam(learning_rate=0.001)
	model.compile(optimizer = opt,
				  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0.1),
				  #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
				  metrics=['accuracy','Recall', 'Precision','FalseNegatives','FalsePositives','CategoricalAccuracy']
				  )
	return model



def create_model_deep(i):
	model = tf.keras.Sequential([
					tf.keras.layers.Dense(i, input_shape=(i,)),
					tf.keras.layers.Dense(i * 2, activation='relu'),
					tf.keras.layers.Dense(i * 1, activation='relu'),
					tf.keras.layers.Dense(i * 1, activation='relu'),
					tf.keras.layers.Dense(3, activation='softmax')
	])
	#opt = tf.keras.optimizers.Adam(learning_rate=0.001)
	opt = tf.keras.optimizers.SGD(learning_rate=0.001)
	model.compile(optimizer = opt,
				  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
				  #loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0.1),
				  #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
				  metrics=['accuracy','Recall', 'Precision','FalseNegatives','FalsePositives','CategoricalAccuracy']
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
	opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
	model.compile(optimizer = opt,
				  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
				  metrics=['accuracy']
				  )
	return model
