"""helpers from data preparation
Author: Feng Liu
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from os import path

import numpy as np 
import scipy.ndimage
import scipy.misc
from tensorflow.python.platform import gfile
from six.moves import urllib


def download_image(source_path, working_dir, filename):
	"""download images from source path to working directory

	Args:
		source_path: where images are stored on server
		working_dir: directory
		filename: downloaded image's filename (0000000, 0000001, ...)
		
	Returns:
		Path to downloaded image	
	"""

	if not gfile.Exists(working_dir):
		gfile.MakeDirs(working_dir)
	filepath = os.path.join(working_dir, filename)
	if not gfile.Exists(filepath):
		temp_file_name, _ = urllib.request.urlretrieve(source_path)
		gfile.Copy(temp_file_name, filepath)
		with gfile.GFile(filepath) as f:
			size = f.size()
		# print('Successfully downloaded', filename, size, 'bytes.')
	return filepath


def parse_lists(map_file):
	"""parse image-label mapping text file

	Args:
		map_file: file where image names and labels are stored
	
	Returns:
		label_list: list of labels
	"""
	image_list = []
	label_list = []
	with open(map_file) as labf:
		for line in labf:
			image = line.rstrip('\n').split('\t')[0]
			label = line.rstrip('\n').split('\t')[1]
			image_list.append(image)
			label_list.append(int(label))
	return image_list, label_list


def make_matrix(image_list, label_list):
	"""convert all images to a numpy matrix
	Args: 
		image_list: list of images on disk
		label_list: list of labels, restored from pair_dict

	Returns:
		X: numpy matrix with shape [data_size, H * W * C]
		y: numpy vector with shape [data_size]
	"""
	# data_size = len([name for name in os.listdir(working_dir) if os.path.isfile(name)])
	data_size = len(label_list)
	y = np.zeros(data_size, dtype = np.int64)
	X = np.zeros([data_size, 180, 180, 3], dtype = np.uint8)

	counter = 0
	for image_path in image_list:
		try:
			image = scipy.ndimage.imread(image_path)
			if image.shape == (180, 180, 3):
				X[counter, :, : ,:] = image 
				y[counter] = label_list[counter]
				counter += 1
			elif len(image.shape) == 3:
				if image.shape[2] == 3:
					image = scipy.misc.imresize(image, (180, 180, 3))
					X[counter, :, : ,:] = image 
					y[counter] = label_list[counter]
					counter += 1
				else:
					print("Image {} is not RBG.".format(image_path))
			else:
				print("Image {} is not RBG.".format(image_path))
		except IOError:
			print("Image {} cannot be read.".format(image_path))
		if counter % 10000 == 0:
			print("{} images processed.".format(counter))

	# trim, if necessary
	if counter < data_size:
		X = X[:counter]
		y = y[:counter]

	return X, y


def split_data(X, y, train_size, val_size=0, test_size=0):
	"""shuffle and split data into train, val, and test set
	Args:
		X: images matrix of shape [data_size, dim]
		y: labels vector of shape [data_size]
		train_size: size of training set
		val_size: size of validation set
		test_seze: size of test set

	Returns: 
		X_train: training images
		y_train: training labels
		X_val: validation images
		y_val: validation labels
		X_test: test images
		y_test: test labels

	Notes: 
		Disused.
	"""
	print("Splitting data...")
	# X = X.reshape((-1, 180, 180, 3))
	data_size = X.shape[0]
	if train_size + val_size + test_size > data_size:
		raise "train_size + val_size + test_size > data_size"
	indices = np.arange(data_size)
	print("Shuffling...")
	np.random.shuffle(indices)

	train_ind = indices[0:train_size]
	val_ind = indices[train_size : train_size + val_size]
	test_ind = indices[train_size + val_size : train_size + val_size + test_size]

	print("Slicing...")
	X_train = X[train_ind]
	y_train = y[train_ind]
	X_val = X[val_ind]
	y_val = y[val_ind]
	X_test = X[test_ind]
	y_test = y[test_ind]

	# normalise to zero-mean: substract mean of training data
	# print("Normalising...")
	# if train_size > 0:
	# 	norm_samp_size = max(train_size, 10000)
	# 	mean_image = np.mean(X_train[:norm_samp_size], axis = 0)
	# 	X_train -= mean_image
	# 	if val_size > 0:
	# 		X_val -= mean_image
	# 	if test_size > 0:
	# 		X_test -= mean_image

	return X_train, y_train, X_val, y_val, X_test, y_test


def prepare_trainval_data(image_list, label_list, train_size, val_size=0):
	"""run this to return data in matrices
	
	Notes:
		Disused.
	"""
	data_size = len(label_list)
	indices = np.arange(data_size)
	np.random.shuffle(indices)
	if train_size + val_size > data_size:
		raise "train_size + val_size > data_size"
	train_ind = indices[0:train_size]
	val_ind = indices[train_size : train_size + val_size]

	val_image_list = [image_list[i] for i in val_ind]
	val_label_list = [label_list[i] for i in val_ind]
	train_image_list = [image_list[i] for i in train_ind]
	train_label_list = [label_list[i] for i in train_ind]

	print("Preparing validation set...")
	X_val, y_val = make_matrix(val_image_list, val_label_list)
	print("Preparing training set...")
	X_train, y_train = make_matrix(train_image_list, train_label_list)

	# X, y = make_matrix(image_list, label_list)
	# X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, train_size, val_size, 0)

	print("X_train.shape:", X_train.shape)
	print("y_train.shape:", y_train.shape)
	print("X_val.shape:", X_val.shape)
	print("y_val.shape:", y_val.shape)

	return X_train, y_train, X_val, y_val


def prepare_test_data(test_image_list):
	"""
	Notes:
		Disused.
	"""
	label_list = [-1] * len(test_image_list)
	print("Preparing test set...")
	X_test, y_test = make_matrix(test_image_list, label_list)
	print("X_test.shape", X_test.shape)

	return X_test

def split_lists(image_list, label_list, train_size, val_size, shuffle=False):
	"""Split image and label list into training and validation set
	   The splitted lists are passed into make_matrix to make 
	   training and validation matrices.

	Args:
		image_list: list of images on disk
		label_list: list of labels, restored from pair_dict
		train_size: size of training set
		val_size: size of validation set

	Returns:
		train_image_list: list of images for making training matrix
		train_label_list: list of labels for making training labels
		val_image_list: list of images for making validation matrix
		val_label_list:	list of labels for making validation labels
	"""
	data_size = len(label_list)
	if train_size + val_size > data_size:
		raise "train_size + val_size > data_size."
	indices = np.arange(data_size)
	if shuffle:
		np.random.shuffle(indices)

	train_ind = indices[0:train_size]
	val_ind = indices[train_size : train_size + val_size]

	val_image_list = [image_list[i] for i in val_ind]
	val_label_list = [label_list[i] for i in val_ind]
	train_image_list = [image_list[i] for i in train_ind]
	train_label_list = [label_list[i] for i in train_ind]

	return train_image_list, train_label_list, val_image_list, val_label_list
