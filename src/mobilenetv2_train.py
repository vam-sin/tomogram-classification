# libraries
from numpy.random import seed
seed(42)
from tensorflow.compat.v1 import set_random_seed 
set_random_seed(42)
from mobilenetv2_model import MobileNetv2
import numpy as np
import mrcfile
import random
from sklearn.preprocessing import normalize
import math
import keras
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
# GPU
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf 

tf.keras.backend.clear_session()

config = ConfigProto()
config.gpu_options.allow_growth = True
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

LIMIT = 4 * 1024
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=LIMIT)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)


classes = ['1bxn', '1f1b', '1yg6', '2byu', '2h12', '2ldb', '3gl1', '3hhb', '4d4r', '6t3e']

def norm_3d(mat):
	res = []
	for i in mat:
		res.append(normalize(i))

	res = np.asarray(res)

	return res 

def pad(x):
	result = np.zeros((32,32,32))
	result[:x.shape[0],:x.shape[1],:x.shape[2]] = x

	return result

def get_train_set(img_per_class):
	files = []
	Y = []
	for i in classes:
		for j in range(img_per_class):
			filename = '../data/SNR003/' + str(i) + '/tomotarget' + str(j) + '.mrc'
			files.append(filename)
			Y.append(classes.index(i))

	# print(files)
	c = list(zip(files, Y))
	random.shuffle(c)
	files, Y = zip(*c)

	count = 0
	while True:
		if count == img_per_class*10:
			count = 0

		f = mrcfile.open(files[count])
		x = f.data
		x = pad(x)
		# x = norm_3d(x)
		x = np.expand_dims(x, axis = 0)
		x = np.expand_dims(x, axis = 4)
		y = Y[count]
		y_arr = np.zeros(10)
		y_arr[y] = 1
		y_arr = np.expand_dims(y_arr, axis = 0)
		# print(x.shape)
		count += 1
		# print(x)

		yield x, y_arr

def get_test_set(img_per_class):
	files = []
	Y = []
	for i in classes:
		for j in range(img_per_class):
			filename = '../data/SNR003/' + str(i) + '/tomotarget' + str(499-j) + '.mrc'
			files.append(filename)
			Y.append(classes.index(i))

	count = 0
	# print(files)
	while True:
		if count == img_per_class*10:
			count = 0

		f = mrcfile.open(files[count])
		x = np.asarray(f.data)
		# x = norm_3d(x)
		x = pad(x)
		x = np.expand_dims(x, axis = 0)
		x = np.expand_dims(x, axis = 4)
		y = Y[count]
		y_arr = np.zeros(10)
		y_arr[y] = 1
		y_arr = np.expand_dims(y_arr, axis = 0)

		count += 1

		yield x, y_arr

train_img = 450
test_img = 500 - train_img 

train_set = get_train_set(train_img)
test_set = get_test_set(test_img)

model = MobileNetv2((32,32,32,1), 10)

opt = keras.optimizers.Adam(learning_rate = 1e-5)

mcp_save = keras.callbacks.callbacks.ModelCheckpoint('models/MobileNetv2.h5', save_best_only=True, monitor='val_accuracy', verbose=1)
reduce_lr = keras.callbacks.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=5, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
callbacks_list = [reduce_lr, mcp_save]

model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
# with tf.device('/cpu:0'):
model.fit_generator(train_set, epochs = 200, steps_per_epoch = train_img*10, verbose=1, validation_data = test_set, validation_steps = test_img*10, callbacks = callbacks_list)

# testing
model = load_model('models/MobileNetv2.h5')
bs = 1

y_pred = []
y_true = []
for i in range(test_img*10):
	x, y_true_sing = next(test_set)
	y_pred_sing = model.predict(x)
	y_pred_sing = np.argmax(y_pred_sing, axis=1)
	# print(y_pred_sing)
	y_pred.append(y_pred_sing[0])

	y_true_sing = np.argmax(y_true_sing, axis=1)
	y_true.append(y_true_sing[0])


print('Confusion Matrix')
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix: ", cm)

print(classification_report(y_true, y_pred))

'''Data
Train: 450
Test: 50 
For each of the 10 classes. 
lr: 1e-4
epochs: 50
bs = 1
'''

'''Results SNR003:

SqueezeNet 3D Convolution Model.
file: squeezenet_noise003.h5
Accuracy:  
'''
