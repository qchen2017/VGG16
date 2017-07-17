#-*-coding:utf-8-*-
"""
train.py: training VGG, save checkpoint, and evaluate on validation set
Feng Liu, liufeng@stanford.edu
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import math
import time
import os

import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile

from util import *
from vgg_net import vgg_net



def do_train(session, pred, loss, image_list, label_list, placeholders,
			 n_epochs=5, batch_size=64, print_every=100, save_every=1, 
			 train_step=None, saver=None, ckpt_dir="", 
			 restore=False, ckpt_path=""):
	"""train model on batches
	
	Args: 
		session: current tf.Session()
		pred: logits, [n_images, n_classes], output from network
		loss: loss function defined with pred and true label
		image_list: list of image paths to train on
		label_list: true labels, associated with image paths
		placeholders: 0: input_placeholder; 1: label_placeholder
		n_epochs: number of epochs to train
		batch_size: size of a batch to compute at one time
		print_every: print information every print_every batches
		save_every: save model as checkpoints every save_every epochs
		train_step: optimiser defined with pred and true label
		saver: tf.train.Saver(), used to save checkpoints
		ckpt_dir: directory to save checkpoints
		restore: if True, then restore from a given checkpoint and continue training
		ckpt_path: checkpoint to restore from and continue training if restore is True

	Returns:
		epoch_loss: training loss evaluated on current epoch
		epoch_correct: training accuracy evaluated on current epoch

	Notes:
		If you would like to restore from checkpoints and continue training,
		please make sure you set shuffle to be False in split_lists() in main().
	"""
	# get placeholders
	input_placeholder = placeholders[0]
	labels_placeholder = placeholders[1]

	# values to calculate
	correct_prediction = tf.equal(tf.argmax(pred, 1), labels_placeholder)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	
	data_size = len(label_list)

	# restore model
	if restore:
		saver.restore(session, ckpt_path)
		print("Model restored from: {}, continue training.".format(ckpt_path))

	# counter
	batch_count = 0
	for e in range(n_epochs):
		print("Now epoch {}:".format(e))
		# shuffle indicies
		
		indices = np.arange(data_size)
		np.random.shuffle(indices)

		# track loss and accuracy
		correct = 0
		losses = []

		# iterate over dataset
		for batch_start in np.arange(0, data_size, batch_size):
			# timing
			start = time.time()

			# slice batch
			batch_indices = indices[batch_start : batch_start + batch_size]
			batch_image_list = [image_list[i] for i in batch_indices]
			batch_label_list = [label_list[i] for i in batch_indices]

			with tf.device('/cpu:0'):
				X_batch, y_batch = make_matrix(batch_image_list, batch_label_list)

			# create feed dict
			feed_dict = {input_placeholder: X_batch,
						 labels_placeholder: y_batch}

			# get batch size
			actual_batch_size = y_batch.shape[0]

			# run session, feed batch, calculate values
			values = [loss, correct_prediction, train_step]
			loss_val, correct_val, _ = session.run(values, feed_dict = feed_dict)

			# aggregate
			losses.append(loss_val * actual_batch_size)
			correct += np.sum(correct_val)

			time_used = time.time() - start
			# print / logging
			if batch_count % print_every == 0:
				print("Batch {0}: with batch training loss = {1:.3g} and accuracy of {2:.2g}" \
					  .format(batch_count, loss_val, np.sum(correct_val) / actual_batch_size))
				print("Time for training one batch: {0:.5g} seconds, average {1:3g} s/image." \
					  .format(time_used, time_used / actual_batch_size))

			batch_count += 1

			
		# accuracy and loss over epoch
		epoch_correct = correct / data_size
		epoch_loss = np.sum(losses) / data_size

		print("Epoch {2}, overall loss = {0:.3g} and accuracy of {1:.3g}" \
			  .format(epoch_loss, epoch_correct, e))
		print("=========================================================")

		# save model
		if saver:
			if not gfile.Exists(ckpt_dir):
				gfile.MakeDirs(ckpt_dir)
			if e % save_every == 0:
				ckpt_path = os.path.join(ckpt_dir, str(e) + ".ckpt")
				ckpt_path = saver.save(session, ckpt_path)
				print("Model saved in: {}".format(ckpt_path))
				print("=========================================================")
    
	return epoch_loss, epoch_correct

def do_evaluate(session, pred, loss, image_list, label_list, placeholders, 
                batch_size=64, restore_from=[0], 
                saver=None, ckpt_dir=""):
	"""evaluate model on validation set

	Args: 
		session: current tf.Session()
		pred: logits, [n_images, n_classes], output from network
		loss: loss function defined with pred and true label
		image_list: list of image paths to evaluate
		label_list: true labels, associated with image paths
		placeholders: 0: input_placeholder; 1: label_placeholder
		batch_size: size of a batch to compute at one time
		restore_from: list of epochs from which model were saved
		saver: tf.train.Saver(), used to restore from checkpoints
		ckpt_dir: directory to restore from checkpoints

	Returns:
		epoch_loss: validation loss evaluated on current epoch
		epoch_correct: validation accuracy evaluated on current epoch

	Notes:
		Wrong predictions are saved in confuse_*.txt with format
			(image_path, true label, predicted label)
	"""
	# get placeholders
	input_placeholder = placeholders[0]
	labels_placeholder = placeholders[1]

	# values to calculate
	correct_prediction = tf.equal(tf.argmax(pred, 1), labels_placeholder)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	data_size = len(label_list)

	for e in restore_from:
		
		# generate indices
		indices = np.arange(data_size)

		# restore model
		if saver:
			ckpt_path = os.path.join(ckpt_dir, str(e) + ".ckpt")
			saver.restore(session, ckpt_path)
			print("Model restored from: {}".format(ckpt_path))

		# track loss, accuracy and prediction results
		correct = 0
		losses = []
		confuse_path = "confuse_{}.txt".format(e)
		with open(confuse_path, 'w+') as f:
			f.write("image\tlabel\tpred\n")

		start = time.time()
		# iterate over dataset
		for batch_start in np.arange(0, data_size, batch_size):
			batch_indices = indices[batch_start : batch_start + batch_size]
      
			# slice batch
			batch_image_list = [image_list[i] for i in batch_indices]
			batch_label_list = [label_list[i] for i in batch_indices]
			with tf.device('/cpu:0'):
				X_batch, y_batch = make_matrix(batch_image_list, batch_label_list)

			# create feed dict
			feed_dict = {input_placeholder: X_batch,
                   		 labels_placeholder: y_batch}

			# get batch size
			actual_batch_size = y_batch.shape[0]

			# run session, feed batch, calculate values
			values = [loss, correct_prediction, tf.argmax(pred, 1)]
			loss_val, correct_val, pred_val = session.run(values, feed_dict = feed_dict) 

			# aggregate
			losses.append(loss_val * actual_batch_size)
			correct += np.sum(correct_val)
			with open(confuse_path, 'a+') as f:
				for i in xrange(0, actual_batch_size): 
					# if correct_val[i] == False:
					if True: 
						f.write(batch_image_list[i] + '\t' + \
								str(batch_label_list[i]) + '\t' + \
								str(pred_val[i]) + '\n')

		time_used = time.time() - start

		epoch_correct = correct / data_size
		epoch_loss = np.sum(losses) / data_size

		print("Overall loss = {0:.3g} and accuracy of {1:.3g}" \
			  .format(epoch_loss, epoch_correct))
		print("Time for evaluation: {0:.5g} seconds, average: {1:.3g} s/image" \
			  .format(time_used, time_used / data_size))

	return epoch_loss, epoch_correct

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument("image_list", 
						help="path to text file of image paths and labels")
	parser.add_argument("n_classes", type=int, 
						help="number of classes, 20 for sample, 2118 for full")
	parser.add_argument("ckpt_dir",
						help="diretory to store checkpoints")
	args = parser.parse_args()

	BATCH_SIZE = 64
	N_EPOCHS = 5
	IMAGE_LIST_PATH = args.image_list
	N_CLASSES = args.n_classes
	CKPT_DIR = args.ckpt_dir

	tf.reset_default_graph()
	
	# Add placeholders
	placeholders = []
	input_placeholder = tf.placeholder(tf.uint8, [None, 180, 180, 3])
	labels_placeholder = tf.placeholder(tf.int64, [None])
	placeholders.append(input_placeholder)
	placeholders.append(labels_placeholder)

	# Build the graph for the deep net
	print("Building graph...")
	y_conv = vgg_net(placeholders[0], N_CLASSES)

	# Define loss and optimizer
	cross_entropy = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(
			labels = tf.one_hot(placeholders[1], N_CLASSES), logits = y_conv))

	train_step = tf.train.AdamOptimizer(5e-4).minimize(cross_entropy)
	
	# Import data
	image_list, label_list = parse_lists(IMAGE_LIST_PATH)

	data_size = len(image_list)
	train_size = int(data_size * 0.9)
	val_size = data_size - train_size

	train_image_list, train_label_list, val_image_list, val_label_list = split_lists(
		image_list, label_list, train_size, val_size, shuffle = False)

	init = tf.global_variables_initializer()
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(init)
		print("Training...")
		do_train(sess, y_conv, cross_entropy, train_image_list, train_label_list, placeholders,
				 n_epochs = N_EPOCHS, batch_size = BATCH_SIZE, print_every = 100, save_every = 1, 
				 train_step = train_step, saver = saver, 
				 ckpt_dir = CKPT_DIR)
		print("Evaluating...")
		do_evaluate(sess, y_conv, cross_entropy, val_image_list, val_label_list, placeholders,
					batch_size = BATCH_SIZE, restore_from = range(0, N_EPOCHS, 1), 
					saver = saver, ckpt_dir = CKPT_DIR)

if __name__ == '__main__':
	# tf.app.run()
	main()
