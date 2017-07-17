#-*-coding:utf-8-*-
"""
predict.py: restore VGG from checkpoint, and make prediction
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

from util import *
from vgg_net import vgg_net



def do_predict(session, pred, image_list, placeholders, 
			   saver, batch_size=64, ckpt_path=""):
	"""predict the product type of images

	Args: 
		session: current tf.Session()
		pred: logits, [n_images, n_classes], output from network
		image_list: list of image paths to predict
		placeholders: input_placeholder
		saver: tf.train.Saver(), used to restore from checkpoints
		batch_size: size of a batch to compute at one time
		ckpt_path: checkpoint path to restore from, eg. ./ckpt/0.ckpt

	Returns:
		results: prediction results
	
	Notes: 
		results will be written to "predictions.txt" if the mode is "default"
	"""
	# get placeholders
	input_placeholder = placeholders[0]

	# prediction
	pred_labels = tf.argmax(pred, 1)
	results = []

	# generate indices
	data_size = len(image_list)
	null_label_list = [-1] * data_size
	indices = np.arange(data_size)

	# restore model
	saver.restore(session, ckpt_path)
	print("Model restored from: {}".format(ckpt_path))

	# iterate over test set
	for batch_start in np.arange(0, data_size, batch_size):
		batch_indices = indices[batch_start : batch_start + batch_size]
		
		# slice batch
		batch_image_list = [image_list[i] for i in batch_indices]
		batch_null_label_list = [null_label_list[i] for i in batch_indices]
		with tf.device('/cpu:0'):
			X_batch, _ = make_matrix(batch_image_list, batch_null_label_list)

		if X_batch.shape[0] == 0:
			raise "Image reading error."
		# create feed dict
		feed_dict = {input_placeholder: X_batch}

		# run session, feed batch, predict
		values = [pred_labels]
		pred_val = session.run(values, feed_dict = feed_dict) 
		for res in pred_val[0]:
			results.append(res)
	
	return results



def main():

	parser = argparse.ArgumentParser()
	parser.add_argument("task", choices=["url", "local", "list"], 
						help="type of task, \"url\", \"local\", or \"list\"")
	parser.add_argument("file_path", 
						help="image url, image local path, or image list path")
	parser.add_argument("n_classes", type=int, 
						help="number of classes, 20 for sample, 2118 for full")
	parser.add_argument("ckpt_path",
						help="path of checkpoint to restore from, eg. \"./ckpt/0.ckpt\"")
	parser.add_argument("-w", "--write", action="store_true",
						help="write result to prediction.txt")
	args = parser.parse_args()

	BATCH_SIZE = 64
	N_EPOCHS = 5
	TASK = args.task
	IMAGE_LIST_PATH = args.file_path
	N_CLASSES = args.n_classes
	CKPT_PATH = args.ckpt_path

	# Import data
	if TASK == "url":
		image_path = download_image(IMAGE_LIST_PATH, "./", "temp.jpg")
		image_list = [image_path]
	elif TASK == "local":
		image_list = [IMAGE_LIST_PATH]
	elif TASK == "list": 
		image_list, _ = parse_lists(IMAGE_LIST_PATH)
	
		
	tf.reset_default_graph()
	
	# Add placeholders
	placeholders = []
	input_placeholder = tf.placeholder(tf.uint8, [None, 180, 180, 3])
	placeholders.append(input_placeholder)

	# Build the graph for the deep net
	print("Building graph...")
	y_conv = vgg_net(placeholders[0], N_CLASSES)

	init = tf.global_variables_initializer()
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(init)
		print("Predicting...")
		results = do_predict(sess, y_conv, image_list, placeholders,
							 batch_size = BATCH_SIZE, saver = saver, 
							 ckpt_path = CKPT_PATH)
		if args.write:
			with open("prediction.txt", 'w+') as f:
				for res in results:
					f.write(str(res) + '\n')
		else:
			print("Predicted product type: {}.".format(results))
	if TASK == "url":
		os.remove(image_path)

if __name__ == '__main__':
	# tf.app.run()
	main()
