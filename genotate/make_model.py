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
				  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
				  #loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0.1),
				  #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
				  metrics=['accuracy','Recall', 'Precision','FalseNegatives','FalsePositives','CategoricalAccuracy']
				  )
	return model



def create_model_deep(i):
	model = tf.keras.Sequential([
					tf.keras.layers.Dense(i, input_shape=(i,)),
					#tf.keras.layers.BatchNormalization(momentum=0.99),
					tf.keras.layers.Dense(i * 2, activation='relu'),
					tf.keras.layers.Dense(i * 1, activation='relu'),
					tf.keras.layers.Dense(i * 1, activation='relu'),
					tf.keras.layers.Dense(3, activation='softmax')
	])
	opt = tf.keras.optimizers.Adam()
	#opt = tf.keras.optimizers.Adam(learning_rate=0.001)
	#opt = tf.keras.optimizers.SGD(learning_rate=0.001)
	model.compile(optimizer = opt,
				  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
				  #loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0.1),
				  #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
				  metrics=['accuracy','Recall', 'Precision','FalseNegatives','FalsePositives']
				  )
	return model

def create_model_conv(args):
	'''
	This creates and returns a new model
	'''
	kreg,reg = (tf.keras.regularizers.l2(0.001),"_l2reg") if args.reg else (None,'')
	model = tf.keras.models.Sequential([
		tf.keras.layers.InputLayer(input_shape=(1,), dtype=tf.string),
		# These are for DNA
		tf.keras.layers.experimental.preprocessing.TextVectorization(
											split = lambda text : tf.strings.unicode_split(text, input_encoding='UTF-8', errors="ignore"),
											max_tokens=6, 
											output_sequence_length=147, 
											vocabulary=['a','c','g','t'] 
											),
		tf.keras.layers.Lambda(lambda x: tf.one_hot(x,depth=6), name='one_hot'),
		tf.keras.layers.Cropping1D(cropping=(args.trim, args.trim)),
		tf.keras.layers.Conv1D(filters=120-(2*args.trim), kernel_size=3, padding='same', activation='relu', kernel_regularizer=kreg, name="conv1" + reg ),
		tf.keras.layers.Conv1D(filters=120-(2*args.trim), kernel_size=3, padding='same', activation='relu', kernel_regularizer=kreg, name="conv2" + reg ),
		#tf.keras.layers.MaxPooling1D(pool_size=3, strides=1),
		# These are for protein
		#tf.keras.layers.experimental.preprocessing.TextVectorization(
		#									split = lambda text : tf.strings.unicode_split(text, input_encoding='UTF-8', errors="ignore"),
		#									standardize=None,
		#									max_tokens=25, 
		#									output_sequence_length=39, 
		#									vocabulary=list('CTSAGPEQKRDNHYFMLVIW*+#') 
		#									),
		#
		#tf.keras.layers.Lambda(lambda x: tf.one_hot(x,depth=25), name='one_hot'),
		#tf.keras.layers.Conv1D(filters=30, kernel_size=5, strides=1, activation='relu'), #, kernel_regularizer=l2_reg(0.001)),
		#tf.keras.layers.MaxPooling1D(pool_size=3, strides=1),
		#tf.keras.layers.Conv1D(filters=30, kernel_size=3, strides=1, activation='relu'), #, kernel_regularizer=l2_reg(0.001)),
		#tf.keras.layers.MaxPooling1D(pool_size=3, strides=1),
		# Done
		#tf.keras.layers.Embedding(6, output_dim=117, mask_zero=True),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(120-(2*args.trim), activation='relu', kernel_regularizer=kreg),
		tf.keras.layers.Dropout(rate=0.05),
		tf.keras.layers.Dense(120-(2*args.trim), activation='relu', kernel_regularizer=kreg),
		tf.keras.layers.Dropout(rate=0.05),
		tf.keras.layers.Dense(3, activation='softmax')
	])
	opt = tf.keras.optimizers.Adam()
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model


def create_model_conv2(args):
	kreg,reg = (tf.keras.regularizers.l2(0.001),"_l2reg") if args.reg else (None,'')
	model = tf.keras.models.Sequential([
		tf.keras.layers.InputLayer(input_shape=(1,), dtype=tf.string),
		tf.keras.layers.experimental.preprocessing.TextVectorization(
											split = lambda text : tf.strings.unicode_split(text, input_encoding='UTF-8', errors="ignore"),
											max_tokens=6, 
											output_sequence_length=147, 
											vocabulary=['a','c','g','t'] 
											),
		tf.keras.layers.Lambda(lambda x: tf.one_hot(x,depth=6), name='one_hot'),
		tf.keras.layers.Cropping1D(cropping=(args.trim, args.trim)),
		#tf.keras.layers.Conv1D(filters=1, kernel_size=3, strides=3, padding='same', activation='relu' ),
		tf.keras.layers.Conv1D(filters=120-(2*args.trim), kernel_size=3, padding='same', activation=args.activation, kernel_regularizer=kreg, name="conv1" + reg ),
		tf.keras.layers.Conv1D(filters=120-(2*args.trim), kernel_size=3, padding='same', activation=args.activation, kernel_regularizer=kreg, name="conv2" + reg ),
		tf.keras.layers.Conv1D(filters=120-(2*args.trim), kernel_size=3, padding='same', activation=args.activation, kernel_regularizer=kreg, name="conv3" + reg ),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(120-(2*args.trim), activation='relu', kernel_regularizer=kreg),
		tf.keras.layers.Dropout(rate=0.05),
		tf.keras.layers.Dense(120-(2*args.trim), activation='relu', kernel_regularizer=kreg),
		tf.keras.layers.Dropout(rate=0.05),
		tf.keras.layers.Dense(120-(2*args.trim), activation='relu', kernel_regularizer=kreg),
		tf.keras.layers.Dropout(rate=0.05),
		tf.keras.layers.Dense(3, activation='softmax')
	])
	opt = tf.keras.optimizers.Adam(learning_rate=0.001)
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy','Recall', 'Precision','FalseNegatives','FalsePositives'])
	return model
