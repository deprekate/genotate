import os

# TensorFlow and tf.keras
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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




def create_model_aa(args):
	'''
	This creates and returns a new model
	'''
	kreg,reg = (tf.keras.regularizers.l2(0.001),"_l2reg") if args.reg else (None,'')
	model = tf.keras.models.Sequential([
		tf.keras.layers.InputLayer(input_shape=(1,), dtype=tf.string),
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


def create_model_dna(args):
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
		# aminoacid space
		#tf.keras.layers.experimental.preprocessing.TextVectorization(split = None, max_tokens=66, output_sequence_length=49,
		#									vocabulary=[a+b+c for a in 'acgt' for b in 'acgt' for c in 'acgt'],
		#									),
		#tf.split(num_or_size_splits=3, axis=1),
		#tf.cast('int32'),
		#tf.strings.unicode_encode(output_encoding='UTF-8'),
		#tf.transpose(),
		#tf.keras.layers.experimental.preprocessing.TextVectorization(split = None,max_tokens=5,output_sequence_length=3,vocabulary=
		#tf.keras.layers.Conv1D(filters=1, kernel_size=3, strides=3, padding='same', activation='relu' ),
		tf.keras.layers.Conv1D(filters=147-(2*args.trim), kernel_size=3, padding='same', activation='relu', kernel_regularizer=kreg, name="conv1" + reg ),
		tf.keras.layers.Conv1D(filters=147-(2*args.trim), kernel_size=3, padding='same', activation='relu', kernel_regularizer=kreg, name="conv2" + reg ),
		tf.keras.layers.Conv1D(filters=147-(2*args.trim), kernel_size=3, padding='same', activation='relu', kernel_regularizer=kreg, name="conv3" + reg ),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(147-(2*args.trim), activation='relu', kernel_regularizer=kreg),
		tf.keras.layers.Dropout(rate=0.05),
		tf.keras.layers.Dense(147-(2*args.trim), activation='relu', kernel_regularizer=kreg),
		tf.keras.layers.Dropout(rate=0.05),
		tf.keras.layers.Dense(147-(2*args.trim), activation='relu', kernel_regularizer=kreg),
		tf.keras.layers.Dropout(rate=0.05),
		tf.keras.layers.Dense(3, activation='softmax')
	])
	opt = tf.keras.optimizers.Adam(learning_rate=0.001)
	custom_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
	model.compile(loss=custom_loss, optimizer=opt, metrics=['accuracy','Recall', 'Precision','FalseNegatives','FalsePositives'])
	#model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

def create_model_blend(args):
	#
	input_ = tf.keras.layers.Input(shape=(1,), dtype=tf.string)
	model_ = tf.keras.layers.experimental.preprocessing.TextVectorization(
										split = lambda text : tf.strings.unicode_split(text, input_encoding='UTF-8', errors="ignore"),
										max_tokens=6, 
										output_sequence_length=99, 
										vocabulary=['a','c','g','t'] 
										)(input_)
	model_ = tf.keras.layers.Lambda(lambda x: tf.one_hot(x,depth=6), name='one_hot')(model_)
	model_ = tf.keras.layers.Conv1D(filters=99, kernel_size=9, padding='same', activation='relu' )(model_)
	model_ = tf.keras.layers.Conv1D(filters=99, kernel_size=9, padding='same', activation='relu' )(model_)
	model_ = tf.keras.layers.Conv1D(filters=99, kernel_size=9, padding='same', activation='relu' )(model_)
	model_ = tf.keras.layers.Flatten()(model_)

	other_ = tf.keras.layers.Input(shape=(3,), dtype=tf.float32)
	#extra_ = tf.keras.layers.Lambda(lambda x: tf.strings.to_number(x), name='strtonum')(other_)

	model_ = tf.keras.layers.concatenate([model_, other_], axis=-1)

	model_ = tf.keras.layers.Dense(99, activation='relu')(model_)
	model_ = tf.keras.layers.Dropout(rate=0.05)(model_)
	model_ = tf.keras.layers.Dense(99, activation='relu')(model_)
	model_ = tf.keras.layers.Dropout(rate=0.05)(model_)
	model_ = tf.keras.layers.Dense(99, activation='relu')(model_)
	model_ = tf.keras.layers.Dropout(rate=0.05)(model_)
	model_ = tf.keras.layers.Dense(3, activation='softmax')(model_)

	merged_model = tf.keras.models.Model([input_, other_], outputs=model_)

	opt = tf.keras.optimizers.Adam(learning_rate=0.001)
	custom_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
	merged_model.compile(loss=custom_loss, optimizer=opt, metrics=['accuracy'])
	#merged_model.compile(loss=custom_loss, optimizer=opt, metrics=['accuracy','Recall', 'Precision','FalseNegatives','FalsePositives'])
	return merged_model

def blend(args):
	input_ = tf.keras.layers.Input(shape=(99,), dtype=tf.float32)
	other_ = tf.keras.layers.Input(shape=(3,), dtype=tf.float32)

	w = tf.cast(input_, dtype=tf.int32)
	w = tf.keras.layers.Lambda(lambda x: tf.one_hot(x,depth=6), name='one_hot')(w)
	w = tf.keras.layers.Conv1D(filters=104, kernel_size=7, padding='same', activation='relu')(w)
	w = tf.keras.layers.Conv1D(filters=104, kernel_size=7, padding='same', activation='relu')(w)
	w = tf.keras.layers.Conv1D(filters=104, kernel_size=7, padding='same', activation='relu')(w)
	w = tf.keras.layers.Flatten()(w)
	w = tf.keras.models.Model(inputs=input_, outputs=w)

	d = tf.keras.layers.Dense(8, activation='relu')(other_)
	d = tf.keras.models.Model(inputs=other_, outputs=d)
	
	combined = tf.keras.layers.concatenate([w.output, d.output]) #, axis=-1)

	z = tf.keras.layers.Dense(88, activation='relu')(combined)
	z = tf.keras.layers.Dropout(rate=0.05)(z)
	z = tf.keras.layers.Dense(88, activation='relu')(z)
	z = tf.keras.layers.Dropout(rate=0.05)(z)
	z = tf.keras.layers.Dense(88, activation='relu')(z)
	z = tf.keras.layers.Dropout(rate=0.05)(z)
	z = tf.keras.layers.Dense(3, activation='softmax')(z)
	
	model = tf.keras.models.Model([w.input, d.input], outputs=z)

	opt = tf.keras.optimizers.Adam(learning_rate=0.001)
	custom_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
	model.compile(loss=custom_loss, optimizer=opt, metrics=['accuracy'])
	return model

def api(args):
	#
	input_ = tf.keras.layers.Input(shape=(87,), dtype=tf.uint8)
	model_ = tf.cast(input_, dtype=tf.int32)
	model_ = tf.keras.layers.Lambda(lambda x: tf.one_hot(x,depth=6), name='one_hot')(model_)
	#model_ = tf.keras.layers.CategoryEncoding(num_tokens=5, output_mode='one_hot')(model_)
	model_ = tf.keras.layers.Conv1D(filters=88, kernel_size=12, padding='same', activation='relu' )(model_)
	model_ = tf.keras.layers.Conv1D(filters=80, kernel_size=15, padding='same', activation='relu' )(model_)
	model_ = tf.keras.layers.Flatten()(model_)
	model_ = tf.keras.layers.Dense( 96, activation='relu')(model_)
	model_ = tf.keras.layers.Dropout(rate=0.03)(model_)
	model_ = tf.keras.layers.Dense(128, activation='relu')(model_)
	model_ = tf.keras.layers.Dropout(rate=0.03)(model_)
	model_ = tf.keras.layers.Dense(112, activation='relu')(model_)
	model_ = tf.keras.layers.Dropout(rate=0.03)(model_)
	model_ = tf.keras.layers.Dense(128, activation='relu')(model_)
	model_ = tf.keras.layers.Dropout(rate=0.03)(model_)
	model_ = tf.keras.layers.Dense( 32, activation='relu')(model_)
	model_ = tf.keras.layers.Dropout(rate=0.03)(model_)
	model_ = tf.keras.layers.Dense(3, activation='softmax')(model_)
	model = tf.keras.models.Model(input_, outputs=model_)
	#lr_fn = tf.keras.optimizers.schedules.InverseTimeDecay(0.005,6633,0.3)
	#lr_fn = tf.keras.optimizers.schedules.ExponentialDecay(0.005,6633,0.97)

	opt = tf.keras.optimizers.legacy.Adam() if hasattr(tf.keras.optimizers, 'legacy') else tf.keras.optimizers.Adam()

	custom_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
	#model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	model.compile(loss=custom_loss, optimizer=opt, metrics=['accuracy']) #,'Recall', 'Precision','FalseNegatives','FalsePositives'])
	return model

'''
import keras_tuner as kt
class HyperRegressor(kt.HyperModel):
	def build(self, hp):
		inputs = tf.keras.layers.Input(shape=(87,), dtype=tf.uint8)
		x = tf.keras.layers.Lambda(lambda x: tf.one_hot(x,depth=6), name='one_hot')(inputs)
		# Tune the number of units in the first Dense layer
		for i in range(hp.Int("conv_layers", 2, 2, default=2)):
			x = tf.keras.layers.Conv1D(
				filters     = hp.Int(f"filters_{i}", 64, 128, step=8, default=88),
				kernel_size = hp.Int(f"kernels_{i}",  6,  20, step=1, default= 7),
				activation  = "relu",
				padding     = "same",
			)(x)
		x = tf.keras.layers.Flatten()(x)
		d = hp.Float("dropout", 0.00, 0.10, step=0.01, default=0.05)
		for i in range(hp.Int("dense_layers", 4, 5, default=5)):
			x = tf.keras.layers.Dense(
				units=hp.Int(f"neurons_{i}", min_value=32, max_value=128, step=16),
				activation='relu'
			)(x)
			x = tf.keras.layers.Dropout(
				rate=d
			)(x)
		outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
	
		model = tf.keras.models.Model(inputs, outputs)
		opt = tf.keras.optimizers.Adam(learning_rate=0.001)
		custom_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
		model.compile(loss=custom_loss, optimizer=opt, metrics=['accuracy'])
		return model

	#def fit(self, hp, model, dataset, **kwargs):
	#	mod = model.evaluate(dataset) 
	#	return mod
'''
