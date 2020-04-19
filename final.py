import numpy as np
import csv

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

from scipy.io import wavfile
import os

from sklearn.decomposition import PCA

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor

training_dir = './data/train'

def read_train_data(train_dir):

	# dictionary  mapping file nubmers to filenames
	filenames = {}
	# dictionary mapping file numbers to calculate 
	fingerprints = {}
	# dictionaries mapping file numbers to waveform data from wav file
	raw_waveforms = {}
	iso_waveforms = {}
	norm_waveforms = {}
	# dictionary mapping file numbers to tuple location
	labels = {}

	for filename in os.listdir(train_dir):

		if filename.endswith('.wav'):

			num_dot_wav = filename.split('_')[-1]
			filenum = int(num_dot_wav.split('.')[0])
			filenames[filenum] = filename

			# init empty list for fingerprints
			fingerprints[filenum] = []

			fs, data = wavfile.read(os.path.join(train_dir,filename))
			raw_waveforms[filenum] = data
			iso_waveforms[filenum] = isolate_audio_pulse(data)
			norm_waveforms[filenum] = isolate_audio_pulse(data)/np.linalg.norm(data)

			# -------------------------------------
			# uncomment below to plot original data
			# and isolated pulse to see difference
			# -------------------------------------
			# fig, axs = plt.subplots(2)
			# fig.suptitle(filename)
			# axs[0].plot(data)
			# axs[1].plot(waveforms[filenum])
			# plt.show()

		elif filename.endswith('.csv'):

			reader = csv.reader(open(os.path.join(train_dir,filename)), delimiter=',')
			for i,row in enumerate(reader):

				if i==0:
					continue

				elif row[0] =='Speaker Location':
					labels[0] = (float(row[1]),float(row[2]))

				else:
					num_dot_wav = row[0].split('_')[-1]
					filenum = int(num_dot_wav.split('.')[0])
					labels[filenum] = (float(row[1]),float(row[2]))

	return filenames, fingerprints, raw_waveforms, iso_waveforms, norm_waveforms, labels

def plot_single_feature(fingerprints, labels, fingerprint_features, feature_index=0):


	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	symbols = ['P','^','x','s','d','p']
	colors = ['blue','green','red','orange','magenta']

	#plot speaker
	x,y = labels[0]
	ax.scatter(x,y,0,marker='x', color = 'red')

	#plot features
	xs, ys, zs = [], [], []
	for i in range(1,len(fingerprints)+1):
		x,y = labels[i]
		xs.append(x)
		ys.append(y)
		zs.append(fingerprints[i][feature_index])
	
	ax.scatter(xs,ys,zs,color='blue')

	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel(fingerprint_features[feature_index])
	plt.show()

def isolate_audio_pulse(waveform):

	peak = np.argmax(waveform)

	return waveform[peak-250:peak+8000]

# Function calculates mean of audio signals
def calc_means(fingerprints, waveforms):

	for i in range(1,len(waveforms)+1):
		fingerprints[i].append(np.mean(waveforms[i]))

# Function calculates mean of absolute value of audio signals
def calc_positive_means(fingerprints, waveforms):

	for i in range(1,len(waveforms)+1):
		fingerprints[i].append(np.mean(np.absolute(waveforms[i])))

# Function calculates mean of absolute value of audio signals
def calc_maxs(fingerprints, waveforms):

	for i in range(1,len(waveforms)+1):
		fingerprints[i].append(np.max(waveforms[i]))

# Function calculates mean of absolute value of audio signals
def calc_stds(fingerprints, waveforms):

	for i in range(1,len(waveforms)+1):
		fingerprints[i].append(np.std(np.absolute(waveforms[i])))

# Function calculates mean of absolute value of audio signals
def calc_vars(fingerprints, waveforms):

	for i in range(1,len(waveforms)+1):
		fingerprints[i].append(np.var(np.absolute(waveforms[i])))

# calculate fft of audio files
def calc_fft(waveforms):

	ffts = {}
	for i in range(1,len(waveforms)+1):
		ffts[i] = np.fft.fft(waveforms[i])
		
		# -------------------------------------
		# uncomment below to plot ffts
		# -------------------------------------
		# print(ffts[i].real.shape)
		# fig, axs = plt.subplots(2)
		# fig.suptitle(i)
		# axs[0].plot(ffts[i].real)
		# axs[1].plot(ffts[i].imag)
		# plt.show()

	return ffts

def train_regressor(fingerprints, labels, train_set):
	max_depth = 30
	regr_multirf = MultiOutputRegressor(RandomForestRegressor(n_estimators=100,
	                                                          max_depth=max_depth,
	                                                          random_state=0))
	X = []
	y = []
	for i in train_set:
		X.append(fingerprints[i])
		y.append(labels[i])

	regr_multirf.fit(X, y)

	return regr_multirf

def eval_regressor(regressor, fingerprints, labels, val_set):
	
	results = {}
	predictions = {}
	for i in val_set:
		x_val = np.asarray(fingerprints[i])
		y_val = regressor.predict(x_val.reshape(1, -1))
		
		results[i] = [
			labels[i], 
			(y_val[0][0],y_val[0][1]),
			(labels[i][0] - y_val[0][0],labels[i][1] - y_val[0][1])
			]

		predictions[i] = (y_val[0][0],y_val[0][1])

		# -------------------------------------
		# uncomment below to print val results
		# -------------------------------------
		# print(f"Example {i} ground gruth: {labels[i]}")
		# print(f"Example {i} prediction : {(y_val[0][0],y_val[0][1])}")
		# print(f"Example {i} difference: {(labels[i][0] - y_val[0][0],labels[i][1] - y_val[0][1])}")
		# print('-'*52)

	return results, predictions

# list for storing feature names 
fingerprint_features = []
filenames, fingerprints, raw_waveforms, iso_waveforms, norm_waveforms, labels = read_train_data(training_dir)

# not quite as clear maybe remove
calc_means(fingerprints, norm_waveforms)
fingerprint_features.append('Means')

calc_positive_means(fingerprints, norm_waveforms)
fingerprint_features.append('Means of Postive waveform')

calc_maxs(fingerprints, norm_waveforms)
fingerprint_features.append('Maxs')

#not quite as clear maybe remove
calc_stds(fingerprints, norm_waveforms)
fingerprint_features.append('Standard Deviations')

calc_vars(fingerprints, norm_waveforms)
fingerprint_features.append('Variance')

#plot_single_feature(fingerprints, labels, fingerprint_features, feature_index=3)

#TODO
# add vectorized window average to fingerprints?
# probably not the best idea, basically blurring the signal
# will remove channel characteristics which is my guess and we want those

ffts = calc_fft(raw_waveforms)

#TODO add fft max?
# ffts_features = calc_fft_features(files,ffts)

# generate permutation of indices for eval and test sets
index_permuation = np.random.permutation(np.arange(1,61))
test_indices, val_indices = index_permuation[:40], index_permuation[41:]

regressor = train_regressor(fingerprints, labels, test_indices)

# output[i] is data of ith example.
# results format: list [ ground truth tuple, prediction tuple, diff tuple]
# prediction format: list [prediction tuple]
results, predictions = eval_regressor(regressor, fingerprints, labels, val_indices)