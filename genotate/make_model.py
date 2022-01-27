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
										output_sequence_length=147, 
										vocabulary=['a','c','g','t'] 
										)(input_)
	model_ = tf.keras.layers.Lambda(lambda x: tf.one_hot(x,depth=6), name='one_hot')(model_)
	model_ = tf.keras.layers.Conv1D(filters=147-(2*args.trim), kernel_size=9, padding='same', activation='relu' )(model_)
	model_ = tf.keras.layers.Conv1D(filters=147-(2*args.trim), kernel_size=9, padding='same', activation='relu' )(model_)
	model_ = tf.keras.layers.Conv1D(filters=147-(2*args.trim), kernel_size=9, padding='same', activation='relu' )(model_)
	model_ = tf.keras.layers.Flatten()(model_)

	other_ = tf.keras.layers.Input(shape=(3,), dtype=tf.string)
	extra_ = tf.keras.layers.Lambda(lambda x: tf.strings.to_number(x), name='strtonum')(other_)

	model_ = tf.keras.layers.concatenate([model_, extra_], axis=-1)

	model_ = tf.keras.layers.Dense(147-(2*args.trim), activation='relu')(model_)
	model_ = tf.keras.layers.Dropout(rate=0.05)(model_)
	model_ = tf.keras.layers.Dense(147-(2*args.trim), activation='relu')(model_)
	model_ = tf.keras.layers.Dropout(rate=0.05)(model_)
	model_ = tf.keras.layers.Dense(147-(2*args.trim), activation='relu')(model_)
	model_ = tf.keras.layers.Dropout(rate=0.05)(model_)
	model_ = tf.keras.layers.Dense(3, activation='softmax')(model_)

	merged_model = tf.keras.models.Model([input_, other_], outputs=model_)

	opt = tf.keras.optimizers.Adam(learning_rate=0.001)
	custom_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
	merged_model.compile(loss=custom_loss, optimizer=opt, metrics=['accuracy','Recall', 'Precision','FalseNegatives','FalsePositives'])
	return merged_model


def create_model_api(args):
	#
	input_ = tf.keras.layers.Input(shape=(1,), dtype=tf.string)
	model_ = tf.keras.layers.experimental.preprocessing.TextVectorization(
										split = lambda text : tf.strings.unicode_split(text, input_encoding='UTF-8', errors="ignore"),
										max_tokens=6, 
										output_sequence_length=147, 
										vocabulary=['a','c','g','t'] 
										)(input_)
	model_ = tf.keras.layers.Lambda(lambda x: tf.one_hot(x,depth=6), name='one_hot')(model_)
	model_ = tf.keras.layers.Conv1D(filters=147, kernel_size=9, padding='same', activation='relu' )(model_)
	model_ = tf.keras.layers.Conv1D(filters=147, kernel_size=9, padding='same', activation='relu' )(model_)
	model_ = tf.keras.layers.Conv1D(filters=147, kernel_size=9, padding='same', activation='relu' )(model_)
	model_ = tf.keras.layers.Flatten()(model_)
	model_ = tf.keras.layers.Dense(147, activation='relu')(model_)
	model_ = tf.keras.layers.Dropout(rate=0.05)(model_)
	model_ = tf.keras.layers.Dense(147, activation='relu')(model_)
	model_ = tf.keras.layers.Dropout(rate=0.05)(model_)
	model_ = tf.keras.layers.Dense(147, activation='relu')(model_)
	model_ = tf.keras.layers.Dropout(rate=0.05)(model_)
	model_ = tf.keras.layers.Dense(3, activation='softmax')(model_)

	model = tf.keras.models.Model(input_, outputs=model_)

	opt = tf.keras.optimizers.Adam(learning_rate=0.001)
	custom_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
	model.compile(loss=custom_loss, optimizer=opt, metrics=['accuracy','Recall', 'Precision','FalseNegatives','FalsePositives'])
	return model
