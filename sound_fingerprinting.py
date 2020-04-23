import os
import csv
import math

import numpy as np
from scipy.io import wavfile

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

def read_data(wav_dir):

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

	for filename in os.listdir(wav_dir):

		if filename.endswith('.wav'):

			num_dot_wav = filename.split('_')[-1]
			filenum = int(num_dot_wav.split('.')[0])
			filenames[filenum] = filename

			# init empty list for fingerprints
			fingerprints[filenum] = []

			fs, data = wavfile.read(os.path.join(wav_dir,filename))
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
			# axs[1].plot(norm_waveforms[filenum])
			# plt.show()

		elif filename.endswith('.csv'):

			reader = csv.reader(open(os.path.join(wav_dir,filename)), delimiter=',')
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

def split_waveforms(waveforms):

	pulses = {}
	echoes = {}

	for i in range(1,len(waveforms)+1):

		peak = np.argmax(waveforms[i])

		pulses[i] = waveforms[i][:(peak+225)] 
		echoes[i] = waveforms[i][(peak+226):]

	return pulses, echoes

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
		ffts[i] = (fft/np.linalg.norm(fft))
		
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
	max_depth = 18
	regr_multirf = MultiOutputRegressor(RandomForestRegressor(n_estimators=200,
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

def fingerprint_prediction(regressor,fingerprints):

	predictions = {}
	for i in range(1, len(fingerprints)+1):
		x_val = np.asarray(fingerprints[i])
		y_val = regressor.predict(x_val.reshape(1, -1))
		
		predictions[i] = (y_val[0][0],y_val[0][1])

	return predictions

def train(training_dir):
	# ******************************************
	# Read in waveform data
	# ------------------------------------------
	# stores names of features for plotting
	fingerprint_features = []
	filenames, fingerprints, raw_waveforms, iso_waveforms, norm_waveforms, labels, fs = read_data(training_dir)

	iso_pulse_waveforms, iso_echo_waveforms = split_waveforms(iso_waveforms)
	pulse_waveforms, echo_waveforms = split_waveforms(norm_waveforms)

	# ******************************************
	# Various waveform Features
	# ------------------------------------------
	# not quite as clear maybe remove
	# calc_means(fingerprints, norm_waveforms)
	# fingerprint_features.append('Means')

	calc_positive_means(fingerprints, norm_waveforms)
	fingerprint_features.append('Means of Postive waveform')

	# calc_signal_energy(fingerprints, iso_waveforms)
	# fingerprint_features.append('calc_signal_energy')

	calc_maxs(fingerprints, norm_waveforms)
	fingerprint_features.append('Maxs')

	#not quite as clear maybe remove
	# calc_stds(fingerprints, norm_waveforms)
	# fingerprint_features.append('Standard Deviations')

	calc_vars(fingerprints, norm_waveforms)
	fingerprint_features.append('Variance')


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
	# Pulse, Echo waveform Features
	# ------------------------------------------
	# not quite as clear maybe remove
	# calc_means(fingerprints, norm_waveforms)
	# fingerprint_features.append('Means')

	calc_positive_means(fingerprints, pulse_waveforms)
	fingerprint_features.append('Means of Postive Pulse')
	calc_positive_means(fingerprints, echo_waveforms)
	fingerprint_features.append('Means of Postive Echo')

	# calc_signal_energy(fingerprints, iso_pulse_waveforms)
	# fingerprint_features.append('Signal Energy of Pulses')
	# calc_signal_energy(fingerprints, iso_echo_waveforms)
	# fingerprint_features.append('Signal Energy of Echo')

	calc_maxs(fingerprints, pulse_waveforms)
	fingerprint_features.append('Pulse Maxs')
	# calc_maxs(fingerprints, echo_waveforms)
	# fingerprint_features.append('Echo Maxs')

	#not quite as clear maybe remove
	calc_stds(fingerprints, pulse_waveforms)
	fingerprint_features.append('Pulse Standard Deviations')
	calc_stds(fingerprints, echo_waveforms)
	fingerprint_features.append('Echo Standard Deviations')

	# calc_vars(fingerprints, pulse_waveforms)
	# fingerprint_features.append('Pulse Variance')
	# calc_vars(fingerprints, echo_waveforms)
	# fingerprint_features.append('Echo Variance')

	# ******************************************
	# Pulse echo FFT Features
	# ------------------------------------------
	# fft of pulses not very meaningful
	# ffts = calc_fft(pulse_waveforms, fs)
	# ffts = calc_fft(echo_waveforms, fs)

	# calc_ffts_max_real(fingerprints, ffts, 5)
	# fingerprint_features.append('max_freq')

	# calc_ffts_max_imag(fingerprints, ffts, 5)
	# fingerprint_features.append('max_phase')

	# ******************************************
	# Plot Specific Feature
	# -----------------------------------------
	# for i in range(len(fingerprint_features)):
	# 	plot_single_feature(fingerprints, labels, fingerprint_features, feature_index=i)


	# ******************************************
	# Regressor Stuff
	# ------------------------------------------
	# ***** UNCOMMENT FOR TUNING ******
	# generate permutation of indices for eval and test sets
	# index_permuation = np.random.permutation(np.arange(1,61))
	# test_indices, val_indices = index_permuation[:42], index_permuation[43:]
	# USE BELOW FOR RUNTIME
	test_indices, val_indices = np.arange(1,61), np.arange(1,61)

	regressor = train_regressor(fingerprints, labels, test_indices)

	# output[i] is data of ith example.
	# results format: list [ ground truth tuple, prediction tuple, diff tuple]
	# prediction format: list [prediction tuple]
	results, predictions = eval_regressor(regressor, fingerprints, labels, val_indices)

	# ******************************************
	# Result Prints
	# ------------------------------------------
	# print (fingerprint_features)
	mean_dis_error = 0.0
	for result in results.items():
		distance = math.sqrt(result[1][2][0] ** 2 + result[1][2][1] **2)
		mean_dis_error += distance

	print ("mean_dis_error", mean_dis_error / len(results.items()))

	return regressor

def test(test_dir, regressor):
	# ******************************************
	# Read in waveform data
	# ------------------------------------------
	# stores names of features for plotting
	fingerprint_features = []
	filenames, fingerprints, raw_waveforms, iso_waveforms, norm_waveforms, _, fs = read_data(test_dir)

	iso_pulse_waveforms, iso_echo_waveforms = split_waveforms(iso_waveforms)
	pulse_waveforms, echo_waveforms = split_waveforms(norm_waveforms)

	# ******************************************
	# Various waveform Features
	# ------------------------------------------
	calc_positive_means(fingerprints, norm_waveforms)
	fingerprint_features.append('Means of Postive waveform')

	calc_maxs(fingerprints, norm_waveforms)
	fingerprint_features.append('Maxs')

	calc_vars(fingerprints, norm_waveforms)
	fingerprint_features.append('Variance')


	# ******************************************
	# FFT Features
	# ------------------------------------------
	ffts = calc_fft(raw_waveforms, fs)

	calc_ffts_max_real(fingerprints, ffts, 5)
	fingerprint_features.append('max_freq')

	calc_ffts_max_imag(fingerprints, ffts, 5)
	fingerprint_features.append('max_phase')

	# ******************************************
	# Pulse, Echo waveform Features
	# ------------------------------------------
	calc_positive_means(fingerprints, pulse_waveforms)
	fingerprint_features.append('Means of Postive Pulse')
	calc_positive_means(fingerprints, echo_waveforms)
	fingerprint_features.append('Means of Postive Echo')

	calc_maxs(fingerprints, pulse_waveforms)
	fingerprint_features.append('Pulse Maxs')

	calc_stds(fingerprints, pulse_waveforms)
	fingerprint_features.append('Pulse Standard Deviations')
	calc_stds(fingerprints, echo_waveforms)
	fingerprint_features.append('Echo Standard Deviations')


	# ******************************************
	# Make predictions
	# ------------------------------------------
	# output[i] is data of ith example.
	# results format: list [ ground truth tuple, prediction tuple, diff tuple]
	# prediction format: list [prediction tuple]
	predictions = fingerprint_prediction(regressor, fingerprints)

	# ******************************************
	# Write Predictions
	# ------------------------------------------
	path = os.path.join(test_dir, "results.txt")
	with open(path, 'w') as file:
		for i in range(1,len(predictions)+1):
			file.write(f'{filenames[i]} prediction: ({predictions[i][0]:.4f},{predictions[i][1]:.4f})\n')

if __name__ == '__main__':
	
	regressor = train('./data/train')

	test('./data/0_Data_RADAR', regressor)
