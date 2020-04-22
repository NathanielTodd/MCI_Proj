import numpy as np
import csv
import math

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

	return filenames, fingerprints, raw_waveforms, iso_waveforms, norm_waveforms, labels, fs

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

	return waveform[peak-250:peak+16000]

# Function calculates mean of audio signals
def calc_means(fingerprints, waveforms):

	for i in range(1,len(waveforms)+1):
		fingerprints[i].append(np.mean(waveforms[i]))

# Function calculates mean of absolute value of audio signals
def calc_positive_means(fingerprints, waveforms):

	for i in range(1,len(waveforms)+1):
		fingerprints[i].append(np.mean(np.absolute(waveforms[i])))

# Function calculates max value of audio signals
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

def calc_signal_energy(fingerprints, waveforms):

	for i in range(1,len(waveforms)+1):
		fingerprints[i].append( np.sum(waveforms[i]**2))

# calculate fft of audio files
def calc_fft(waveforms, fs):

	ffts = {}
	for i in range(1,len(waveforms)+1):
		fft = np.fft.fft(waveforms[i])
		ffts[i] = (fft/np.linalg.norm(fft))[20000:95000]
		
		# -------------------------------------
		# uncomment below to plot ffts
		# -------------------------------------
		# print(ffts[i].real.shape, fs)
		# fig, axs = plt.subplots(2)
		# fig.suptitle(i)
		# axs[0].plot(ffts[i].real)
		# axs[1].plot(ffts[i].imag)
		# plt.show()

	return ffts

def fft_real_fit_PCA(ffts,n):
	pca = PCA(n_components=n)

	X = []
	for i in range(1, len(ffts) + 1):
		fft = ffts[i].real
		print(fft.shape)
		X.append(fft)

	return pca.fit(np.stack(X))

def fft_imag_fit_PCA(ffts,n):
	pca = PCA(n_components=n)

	X = []
	for i in range(1, len(ffts) + 1):
		X.append(ffts[i].imag)

	return pca.fit(np.stack(X))

def fft_real_dim_reduce(fingerprints, ffts, pca):

	for i in range(1, len(ffts) + 1):
		fingerprints[i] += list(pca.transform(ffts[i].real.reshape(1,-1)).flatten())

def fft_imag_dim_reduce(fingerprints, ffts, pca):

	for i in range(1, len(ffts) + 1):
		fingerprints[i] += list(pca.transform(ffts[i].imag.reshape(1,-1)).flatten())

def calc_ffts_max_real(fingerprints, ffts,n):

	for i in range(1, len(ffts) + 1):
		max_freqs = np.argsort(np.absolute(ffts[i].real))
		fingerprints[i] += list(max_freqs[0:n])

def calc_ffts_max_imag(fingerprints, ffts,n):

	for i in range(1, len(ffts) + 1):
		max_phases = np.argsort(np.absolute(ffts[i].imag))
		fingerprints[i] += list(max_phases[0:n])

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
		print(f"Example {i} ground gruth: {labels[i]}")
		print(f"Example {i} prediction : ({y_val[0][0]:.4f},{y_val[0][1]:.4f})")
		print(f"Example {i} difference: ({labels[i][0] - y_val[0][0]:.4f},{labels[i][1] - y_val[0][1]:.4f})")
		print('-'*52)

	return results, predictions

# use KNeighborsRegressor 
def train_knn_regressor(fingerprints, labels, train_set):
	neigh = KNeighborsRegressor(n_neighbors=3)

	X = []
	y = []
	for i in train_set:
		X.append(fingerprints[i])
		y.append(labels[i])

	neigh.fit(X, y)

	return neigh

def eval_knn_regressor(regressor, fingerprints, labels, val_set):
	results = {}
	predictions = {}
	for i in val_set:
		x_val = np.asarray(fingerprints[i])
		y_val = regressor.kneighbors(x_val.reshape(1, -1))
		y_val_x_mean = 0.0
		y_val_y_mean = 0.0
		for val in y_val[1]:
			for v, i in enumerate(val):
				y_val_x_mean += labels[v][0]
				y_val_y_mean += labels[v][1]
		y_val_x_mean /= 3
		y_val_y_mean /= 3

		results[i] = [
			labels[i], 
			(y_val_x_mean,y_val_y_mean),
			(labels[i][0] - y_val_x_mean,labels[i][1] - y_val_y_mean)
			]

		predictions[i] = (y_val_x_mean,y_val_y_mean)

	return results, predictions

# list for storing feature names 
fingerprint_features = []
filenames, fingerprints, raw_waveforms, iso_waveforms, norm_waveforms, labels, fs = read_train_data(training_dir)

# ******************************************
# Vairous Features
# ------------------------------------------
# not quite as clear maybe remove
# calc_means(fingerprints, norm_waveforms)
# fingerprint_features.append('Means')

calc_positive_means(fingerprints, norm_waveforms)
fingerprint_features.append('Means of Postive waveform')

calc_maxs(fingerprints, norm_waveforms)
fingerprint_features.append('Maxs')

#not quite as clear maybe remove
calc_stds(fingerprints, norm_waveforms)
fingerprint_features.append('Standard Deviations')

calc_vars(fingerprints, norm_waveforms)
fingerprint_features.append('Variance')

calc_signal_energy(fingerprints, iso_waveforms)
fingerprint_features.append('calc_signal_energy')

# ******************************************
# FFT Features
# ------------------------------------------
ffts = calc_fft(raw_waveforms, fs)

calc_ffts_max_real(fingerprints, ffts, 5)
fingerprint_features.append('max_freq')

calc_ffts_max_imag(fingerprints, ffts, 5)
fingerprint_features.append('max_phase')

#PCA with ffts
# fft_real_pca = fft_real_fit_PCA(ffts,3)
# fft_imag_pca = fft_imag_fit_PCA(ffts,3)

# fft_real_dim_reduce(fingerprints, ffts, fft_real_pca)
# fingerprint_features.append('ffts_real_dim_reduction')

# fft_imag_dim_reduce(fingerprints, ffts, fft_imag_pca)
# fingerprint_features.append('ffts_imag_dim_reduction')


# ******************************************
# Plot Specific Feature
# -----------------------------------------4
# plot_single_feature(fingerprints, labels, fingerprint_features, feature_index=4)


# ******************************************
# Regressor Stuff
# ------------------------------------------
# generate permutation of indices for eval and test sets
index_permuation = np.random.permutation(np.arange(1,61))
test_indices, val_indices = index_permuation[:42], index_permuation[43:]

regressor = train_regressor(fingerprints, labels, test_indices)

# output[i] is data of ith example.
# results format: list [ ground truth tuple, prediction tuple, diff tuple]
# prediction format: list [prediction tuple]
results, predictions = eval_regressor(regressor, fingerprints, labels, val_indices)

# use Knn regressor 
# knn_regressor = train_knn_regressor(fingerprints, labels, test_indices)
# knn_results, knn_predictions = eval_knn_regressor(knn_regressor, fingerprints, labels, val_indices)


# ******************************************
# Result Prints
# ------------------------------------------
# print (fingerprint_features)
mean_dis_error = 0.0
for result in results.items():
	distance = math.sqrt(result[1][2][0] ** 2 + result[1][2][1] **2)
	mean_dis_error += distance

print ("mean_dis_error", mean_dis_error / len(results.items()))

# mean_dis_error = 0.0
# for result in knn_results.items():
# 	 print (result)
#	 distance = math.sqrt(result[1][2][0] ** 2 + result[1][2][1] **2)
#	 mean_dis_error += distance

# print ("knn_mean_dis_error", mean_dis_error / len(results.items()))