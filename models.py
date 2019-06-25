import tensorflow as tf
import modules_tf as modules
import config
from data_pipeline import data_gen_pho, data_gen_full
import time, os
import utils
import h5py
import numpy as np
import mir_eval
import pandas as pd
from random import randint
import librosa
# import sig_process
import matplotlib.pyplot as plt
from scipy.ndimage import filters

class Model(object):
	def __init__(self):
		self.get_placeholders()
		self.model()


	def test_file_all(self, file_name, sess):
		"""
		Function to extract multi pitch from file. Currently supports only HDF5 files.
		"""
		scores = self.extract_f0_file(file_name, sess)
		return scores

	def validate_file(self, file_name, sess):
		"""
		Function to extract multi pitch from file, for validation. Currently supports only HDF5 files.
		"""
		scores = self.extract_f0_file(file_name, sess)
		pre = scores['Precision']
		acc = scores['Accuracy']
		rec = scores['Recall']
		return pre, acc, rec




	def load_model(self, sess, log_dir):
		"""
		Load model parameters, for synthesis or re-starting training. 
		"""
		self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		self.saver = tf.train.Saver(max_to_keep= config.max_models_to_keep)


		sess.run(self.init_op)

		ckpt = tf.train.get_checkpoint_state(log_dir)

		if ckpt and ckpt.model_checkpoint_path:
			print("Using the model in %s"%ckpt.model_checkpoint_path)
			self.saver.restore(sess, ckpt.model_checkpoint_path)


	def save_model(self, sess, epoch, log_dir):
		"""
		Save the model.
		"""
		checkpoint_file = os.path.join(log_dir, 'model.ckpt')
		self.saver.save(sess, checkpoint_file, global_step=epoch)

	def print_summary(self, print_dict, epoch, duration):
		"""
		Print training summary to console, every N epochs.
		Summary will depend on model_mode.
		"""

		print('epoch %d took (%.3f sec)' % (epoch + 1, duration))
		for key, value in print_dict.items():
			print('{} : {}'.format(key, value))
			
class Phone_Net(Model):

	def loss_function(self):
		"""
		returns the loss function for the model, based on the mode. 
		"""
		# self.final_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="Final_Model")

		# self.g_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="Generator")

		# self.d_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="Discriminator")

		# self.phone_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="phone_Model")



		# Phoneme network loss and summary


		self.pho_weights = tf.reduce_sum(config.phonemas_weights * self.phone_onehot_labels, axis=-1)

		self.unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.phone_onehot_labels, logits = self.pho_logits)

		self.weighted_losses = self.unweighted_losses * self.pho_weights

		self.pho_loss = tf.reduce_mean(self.weighted_losses)

		self.pho_acc = tf.metrics.accuracy(labels = self.phoneme_labels, predictions = self.pho_classes)


	def get_optimizers(self):
		"""
		Returns the optimizers for the model, based on the loss functions and the mode. 
		"""
		self.optimizer = tf.train.AdamOptimizer(learning_rate = config.init_lr)
		self.global_step = tf.Variable(0, name='global_step', trainable=False)
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			self.train_function = self.optimizer.minimize(self.pho_loss, global_step = self.global_step)

	def get_summary(self, sess, log_dir):
		"""
		Gets the summaries and summary writers for the losses.
		"""

		self.pho_summary = tf.summary.scalar('pho_loss', self.pho_loss)

		self.pho_acc_summary = tf.summary.scalar('pho_accuracy', self.pho_acc[0])


		self.train_summary_writer = tf.summary.FileWriter(log_dir+'train/', sess.graph)
		self.val_summary_writer = tf.summary.FileWriter(log_dir+'val/', sess.graph)
		self.summary = tf.summary.merge_all()

	def get_placeholders(self):
		"""
		Returns the placeholders for the model. 
		Depending on the mode, can return placeholders for either just the generator or both the generator and discriminator.
		"""

		self.input_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size, config.max_phr_len, config.input_features),
										   name='input_placeholder')

		self.phoneme_labels = tf.placeholder(tf.int32, shape=(config.batch_size, config.max_phr_len),
										name='phoneme_placeholder')
		self.phone_onehot_labels = tf.one_hot(indices=tf.cast(self.phoneme_labels, tf.int32), depth=42)

		self.is_train = tf.placeholder(tf.bool, name="is_train")


	def read_input_file(self, file_name):
		"""
		Function to read and process input file, given name and the synth_mode.
		Returns features for the file based on mode (0 for hdf5 file, 1 for wav file).
		Currently, only the HDF5 version is implemented.
		"""
		# if file_name.endswith('.hdf5'):
		feat_file = h5py.File(config.feats_dir + file_name)
		atb = feat_file['atb'][()]

		# atb = atb[:, 1:]

		hcqt = feat_file['voc_hcqt'][()]

		feat_file.close()

		in_batches_hcqt, nchunks_in = utils.generate_overlapadd(hcqt.reshape(-1,6*360))
		in_batches_hcqt = in_batches_hcqt.reshape(in_batches_hcqt.shape[0], config.batch_size, config.max_phr_len,
												  6, 360)
		in_batches_hcqt = np.swapaxes(in_batches_hcqt, -1, -2)
		return in_batches_hcqt, atb, nchunks_in

	def read_input_wav_file(self, file_name):
		"""
		Function to read and process input file, given name and the synth_mode.
		Returns features for the file based on mode (0 for hdf5 file, 1 for wav file).
		Currently, only the HDF5 version is implemented.
		"""
		audio, fs = librosa.core.load(file_name, sr=config.fs)
		hcqt = sig_process.get_hcqt(audio/4)

		hcqt = np.swapaxes(hcqt, 0, 1)

		in_batches_hcqt, nchunks_in = utils.generate_overlapadd(hcqt.reshape(-1,6*360))
		in_batches_hcqt = in_batches_hcqt.reshape(in_batches_hcqt.shape[0], config.batch_size, config.max_phr_len,
												  6, 360)
		in_batches_hcqt = np.swapaxes(in_batches_hcqt, -1, -2)

		return in_batches_hcqt, nchunks_in, hcqt.shape[0]




	def test_file(self, file_name):
		"""
		Function to extract multi pitch from file. Currently supports only HDF5 files.
		"""
		sess = tf.Session()
		self.load_model(sess, log_dir = config.log_dir)
		scores = self.extract_f0_file(file_name, sess)
		return scores

	def test_wav_file(self, file_name, save_path):
		"""
		Function to extract multi pitch from wav file.
		"""

		sess = tf.Session()
		self.load_model(sess, log_dir = config.log_dir)
		in_batches_hcqt, nchunks_in, max_len = self.read_input_wav_file(file_name)
		out_batches_atb = []
		for in_batch_hcqt in in_batches_hcqt:
			feed_dict = {self.input_placeholder: in_batch_hcqt, self.is_train: False}
			out_atb = sess.run(self.outputs, feed_dict=feed_dict)
			out_batches_atb.append(out_atb)
		out_batches_atb = np.array(out_batches_atb)
		out_batches_atb = utils.overlapadd(out_batches_atb.reshape(out_batches_atb.shape[0], config.batch_size, config.max_phr_len, -1),
						 nchunks_in)
		out_batches_atb = out_batches_atb[:max_len]
		# plt.imshow(out_batches_atb.T, origin = 'lower', aspect = 'auto')
		#
		# plt.show()
		# import pdb;pdb.set_trace()

		time_1, ori_freq = utils.process_output(out_batches_atb)
		utils.save_multif0_output(time_1, ori_freq, save_path)





	def train(self):
		"""
		Function to train the model, and save Tensorboard summary, for N epochs. 
		"""
		sess = tf.Session()


		self.loss_function()
		self.get_optimizers()
		self.load_model(sess, config.log_dir)
		self.get_summary(sess, config.log_dir)
		start_epoch = int(sess.run(tf.train.get_global_step()) / (config.batches_per_epoch_train))


		print("Start from: %d" % start_epoch)


		for epoch in range(start_epoch, config.num_epochs):
			data_generator = data_gen_pho()
			val_generator = data_gen_pho(mode = 'Val')
			start_time = time.time()


			batch_num = 0
			epoch_train_loss = 0
			epoch_train_acc = 0
			epoch_val_loss = 0
			epoch_val_acc = 0

			with tf.variable_scope('Training'):
				for spec, phons in data_generator:

					step_loss, step_acc, summary_str = self.train_model(spec, phons, sess)
					epoch_train_loss+=step_loss
					epoch_train_acc+=step_acc

					self.train_summary_writer.add_summary(summary_str, epoch)
					self.train_summary_writer.flush()

					utils.progress(batch_num,config.batches_per_epoch_train, suffix = 'training done')

					batch_num+=1

				epoch_train_loss = epoch_train_loss/batch_num
				epoch_train_acc = epoch_train_acc/batch_num
				print_dict = {"Training Loss": epoch_train_loss}
				print_dict["Training Accuracy"] =  epoch_train_acc

			if (epoch + 1) % config.validate_every == 0:
				batch_num = 0
				with tf.variable_scope('Validation'):
					for spec, phons in val_generator:
						step_loss, step_acc, summary_str = self.validate_model(spec, phons, sess)
						epoch_val_loss += step_loss
						epoch_val_acc += step_acc

						self.val_summary_writer.add_summary(summary_str, epoch)
						self.val_summary_writer.flush()
						batch_num+=1

						utils.progress(batch_num, config.batches_per_epoch_val, suffix='validation done')

					epoch_val_loss = epoch_val_loss / batch_num
					epoch_val_acc = epoch_val_acc / batch_num
					print_dict["Validation Loss"] = epoch_val_loss
					print_dict["Validation Accuracy"] = epoch_val_acc


			end_time = time.time()
			if (epoch + 1) % config.print_every == 0:
				self.print_summary(print_dict, epoch, end_time-start_time)
			if (epoch + 1) % config.save_every == 0 or (epoch + 1) == config.num_epochs:
				self.save_model(sess, epoch+1, config.log_dir)

	def train_model(self, spec, phons, sess):
		"""
		Function to train the model for each epoch
		"""
		feed_dict = {self.input_placeholder: spec, self.phoneme_labels: phons, self.is_train: True}
		_, step_loss, step_acc = sess.run(
			[self.train_function, self.pho_loss, self.pho_acc], feed_dict=feed_dict)
		summary_str = sess.run(self.summary, feed_dict=feed_dict)

		return step_loss, step_acc[0], summary_str

	def validate_model(self,spec, phons, sess):
		"""
		Function to train the model for each epoch
		"""
		feed_dict = {self.input_placeholder: spec, self.phoneme_labels: phons, self.is_train: False}

		step_loss, step_acc = sess.run([self.pho_loss, self.pho_acc], feed_dict=feed_dict)
		summary_str = sess.run(self.summary, feed_dict=feed_dict)
		return step_loss, step_acc[0], summary_str


	def eval_all(self, file_name_csv):
		sess = tf.Session()
		self.load_model(sess, config.log_dir)
		val_list = config.val_list
		count = 0
		scores = {}
		for file_name in val_list:
			file_score = self.test_file_all(file_name, sess)
			if count == 0:
				for key, value in file_score.items():
					scores[key] = [value]
				scores['file_name'] = [file_name]
			else:
				for key, value in file_score.items():
					scores[key].append(value)
				scores['file_name'].append(file_name)

				# import pdb;pdb.set_trace()
			count += 1
			utils.progress(count, len(val_list), suffix='validation done')
		utils.save_scores_mir_eval(scores, file_name_csv)

		return scores


	def model(self):
		"""
		The main model function, takes and returns tensors.
		Defined in modules.

		"""
		with tf.variable_scope('phone_Model') as scope:
			# regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
			self.pho_logits = modules.phone_network(self.input_placeholder, self.is_train)
			self.pho_classes = tf.argmax(self.pho_logits, axis=-1)
			self.pho_probs = tf.nn.softmax(self.pho_logits)

class MultiSynth(Model):

	def get_optimizers(self):
		"""
		Returns the optimizers for the model, based on the loss functions and the mode. 
		"""

		self.pho_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'Phone_Model')
		self.singer_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'Singer_Model')
		self.f0_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'F0_Model')
		self.final_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'Final_Model')

		self.pho_optimizer = tf.train.AdamOptimizer(learning_rate = config.init_lr)
		self.final_optimizer = tf.train.AdamOptimizer(learning_rate = config.init_lr)
		self.singer_optimizer = tf.train.AdamOptimizer(learning_rate = config.init_lr)
		self.f0_optimizer = tf.train.AdamOptimizer(learning_rate = config.init_lr)

		self.global_step = tf.Variable(0, name='global_step', trainable=False)
		self.global_step_pho = tf.Variable(0, name='pho_global_step', trainable=False)
		self.global_step_f0 = tf.Variable(0, name='f0_global_step', trainable=False)
		self.global_step_singer = tf.Variable(0, name='singer_global_step', trainable=False)


		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			self.final_train_function = self.final_optimizer.minimize(self.final_loss, global_step = self.global_step, var_list = self.final_params)
			self.pho_train_function = self.pho_optimizer.minimize(self.pho_loss, global_step = self.global_step_pho, var_list = self.pho_params)
			self.f0_train_function = self.f0_optimizer.minimize(self.f0_loss, global_step = self.global_step_f0, var_list = self.f0_params)
			self.singer_train_function = self.singer_optimizer.minimize(self.singer_loss, global_step = self.global_step_singer, var_list = self.singer_params)

	def loss_function(self):
		"""
		returns the loss function for the model, based on the mode. 
		"""

		self.pho_weights = tf.reduce_sum(config.phonemas_weights * self.phone_onehot_labels, axis=-1)

		self.unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.phone_onehot_labels, logits = self.pho_logits)

		self.weighted_losses = self.unweighted_losses * self.pho_weights

		self.pho_loss = tf.reduce_mean(self.weighted_losses)

		self.pho_acc = tf.metrics.accuracy(labels = self.phoneme_labels, predictions = self.pho_classes)

		self.singer_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.singer_onehot_labels, logits=self.singer_logits))

		self.singer_acc = tf.metrics.accuracy(labels=self.singer_labels , predictions=self.singer_classes)

		self.f0_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.f0_onehot_labels, logits=self.f0_logits))

		self.f0_acc = tf.metrics.accuracy(labels=self.f0_labels , predictions=self.f0_classes)

		self.final_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels= self.input_placeholder, logits = self.output)) /(config.batch_size*config.max_phr_len*config.output_features)

	def get_summary(self, sess, log_dir):
		"""
		Gets the summaries and summary writers for the losses.
		"""

		self.pho_summary = tf.summary.scalar('pho_loss', self.pho_loss)

		self.pho_acc_summary = tf.summary.scalar('pho_accuracy', self.pho_acc[0])

		self.final_summary = tf.summary.scalar('final_loss', self.final_loss)

		self.f0_summary = tf.summary.scalar('f0_loss', self.f0_loss)

		self.f0_acc_summary = tf.summary.scalar('f0_accuracy', self.f0_acc[0])

		self.singer_summary = tf.summary.scalar('singer_loss', self.singer_loss)

		self.singer_acc_summary = tf.summary.scalar('singer_accuracy', self.singer_acc[0])

		self.train_summary_writer = tf.summary.FileWriter(log_dir+'train/', sess.graph)
		self.val_summary_writer = tf.summary.FileWriter(log_dir+'val/', sess.graph)
		self.summary = tf.summary.merge_all()

	def get_placeholders(self):
		"""
		Returns the placeholders for the model. 
		Depending on the mode, can return placeholders for either just the generator or both the generator and discriminator.
		"""

		self.input_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size, config.max_phr_len*2**8),
										   name='input_placeholder')

		self.phoneme_labels_blur = tf.placeholder(tf.int32, shape=(config.batch_size, config.max_phr_len, config.num_phos),
										name='phoneme_blur_placeholder')

		self.phoneme_labels = tf.placeholder(tf.int32, shape=(config.batch_size, config.max_phr_len),
										name='phoneme_placeholder')
		self.phone_onehot_labels = tf.one_hot(indices=tf.cast(self.phoneme_labels, tf.int32), depth = config.num_phos)
		
		self.f0_labels = tf.placeholder(tf.int32, shape=(config.batch_size, config.max_phr_len),
										name='f0_placeholder')
		self.f0_onehot_labels = tf.one_hot(indices=tf.cast(self.f0_labels, tf.int32), depth = config.num_f0)

		self.singer_labels = tf.placeholder(tf.float32, shape=(config.batch_size),name='singer_placeholder')
		self.singer_onehot_labels = tf.one_hot(indices=tf.cast(self.singer_labels, tf.int32), depth = config.num_singers)

		self.is_train = tf.placeholder(tf.bool, name="is_train")


	def train(self):
		"""
		Function to train the model, and save Tensorboard summary, for N epochs. 
		"""
		sess = tf.Session()


		self.loss_function()
		self.get_optimizers()
		self.load_model(sess, config.log_dir)
		self.get_summary(sess, config.log_dir)
		start_epoch = int(sess.run(tf.train.get_global_step()) / (config.batches_per_epoch_train))


		print("Start from: %d" % start_epoch)


		for epoch in range(start_epoch, config.num_epochs):
			data_generator = data_gen_full()
			val_generator = data_gen_full(mode = 'Val')
			start_time = time.time()


			batch_num = 0
			epoch_final_loss = 0

			epoch_pho_loss = 0
			epoch_pho_acc = 0

			epoch_f0_loss = 0
			epoch_f0_acc = 0

			epoch_singer_loss = 0
			epoch_singer_acc = 0

			val_final_loss = 0

			val_pho_loss = 0
			val_pho_acc = 0

			val_f0_loss = 0
			val_f0_acc = 0

			val_singer_loss = 0
			val_singer_acc = 0

			with tf.variable_scope('Training'):
				for mix_in, pho_targs, f0_targs, singer_targs in data_generator:

					final_loss, pho_loss, pho_acc,f0_loss, f0_acc, singer_loss, singer_acc, summary_str = self.train_model(mix_in, pho_targs, f0_targs, singer_targs, sess)

					# import pdb;pdb.set_trace()

					epoch_final_loss+=final_loss

					epoch_pho_loss+=pho_loss
					epoch_pho_acc+=pho_acc

					epoch_f0_loss+=f0_loss
					epoch_f0_acc+=f0_acc

					epoch_singer_loss+=singer_loss
					epoch_singer_acc+=singer_acc

					self.train_summary_writer.add_summary(summary_str, epoch)
					self.train_summary_writer.flush()

					utils.progress(batch_num,config.batches_per_epoch_train, suffix = 'training done')

					batch_num+=1

				epoch_final_loss = epoch_final_loss/batch_num

				epoch_pho_loss = epoch_pho_loss/batch_num
				epoch_pho_acc = epoch_pho_acc/batch_num

				epoch_f0_loss = epoch_f0_loss/batch_num
				epoch_f0_acc = epoch_f0_acc/batch_num

				epoch_singer_loss = epoch_singer_loss/batch_num
				epoch_singer_acc = epoch_singer_acc/batch_num

				print_dict = {"Final Loss": epoch_final_loss}

				print_dict["Pho Loss"] =  epoch_pho_loss
				print_dict["Pho Accuracy"] =  epoch_pho_acc

				print_dict["F0 Loss"] =  epoch_f0_loss
				print_dict["F0 Accuracy"] =  epoch_f0_acc

				print_dict["Singer Loss"] =  epoch_singer_loss
				print_dict["Singer Accuracy"] =  epoch_singer_acc

			if (epoch + 1) % config.validate_every == 0:
				batch_num = 0
				with tf.variable_scope('Validation'):
					for mix_in, pho_targs, f0_targs, singer_targs in val_generator:

						final_loss, pho_loss, pho_acc,f0_loss, f0_acc, singer_loss, singer_acc, summary_str = self.validate_model(mix_in, pho_targs, f0_targs, singer_targs, sess)
						val_final_loss+=final_loss

						val_pho_loss+=pho_loss
						val_pho_acc+=pho_acc

						val_f0_loss+=f0_loss
						val_f0_acc+=f0_acc

						val_singer_loss+=singer_loss
						val_singer_acc+=singer_acc

						self.val_summary_writer.add_summary(summary_str, epoch)
						self.val_summary_writer.flush()
						batch_num+=1

						utils.progress(batch_num, config.batches_per_epoch_val, suffix='validation done')

					val_final_loss = val_final_loss/batch_num

					val_pho_loss = val_pho_loss/batch_num
					val_pho_acc = val_pho_acc/batch_num

					val_f0_loss = val_f0_loss/batch_num
					val_f0_acc = val_f0_acc/batch_num

					val_singer_loss = val_singer_loss/batch_num
					val_singer_acc = val_singer_acc/batch_num\

					print_dict["Val Final Loss"] =  val_final_loss

					print_dict["Val Pho Loss"] =  val_pho_loss
					print_dict["Val Pho Accuracy"] =  val_pho_acc

					print_dict["Val F0 Loss"] =  val_f0_loss
					print_dict["Val F0 Accuracy"] =  val_f0_acc

					print_dict["Val Singer Loss"] =  val_singer_loss
					print_dict["Val Singer Accuracy"] =  val_singer_acc

			end_time = time.time()
			if (epoch + 1) % config.print_every == 0:
				self.print_summary(print_dict, epoch, end_time-start_time)
			if (epoch + 1) % config.save_every == 0 or (epoch + 1) == config.num_epochs:
				self.save_model(sess, epoch+1, config.log_dir)

	def train_model(self, mix_in, pho_targs, f0_targs, singer_targs, sess):
		"""
		Function to train the model for each epoch
		"""
		feed_dict = {self.input_placeholder: mix_in, self.phoneme_labels: pho_targs, self.f0_labels: f0_targs, self.singer_labels: singer_targs, self.is_train: True}

		_, _, _, _, final_loss, pho_loss, pho_acc,f0_loss, f0_acc, singer_loss, singer_acc = sess.run(
			[self.f0_train_function,self.pho_train_function, self.singer_train_function, self.final_train_function, self.final_loss, self.pho_loss,
			 self.pho_acc, self.f0_loss, self.f0_acc, self.singer_loss, self.singer_acc], feed_dict=feed_dict)

		summary_str = sess.run(self.summary, feed_dict=feed_dict)

		return final_loss, pho_loss, pho_acc[0],f0_loss, f0_acc[0], singer_loss, singer_acc[0],  summary_str

	def validate_model(self,mix_in, pho_targs, f0_targs, singer_targs, sess):
		"""
		Function to train the model for each epoch
		"""
		feed_dict = {self.input_placeholder: mix_in, self.phoneme_labels: pho_targs, self.f0_labels: f0_targs, self.singer_labels: singer_targs, self.is_train: False}

		final_loss, pho_loss, pho_acc,f0_loss, f0_acc, singer_loss, singer_acc = sess.run(
			[self.final_loss, self.pho_loss,
			 self.pho_acc, self.f0_loss, self.f0_acc, self.singer_loss, self.singer_acc], feed_dict=feed_dict)

		summary_str = sess.run(self.summary, feed_dict=feed_dict)

		return final_loss, pho_loss, pho_acc[0],f0_loss, f0_acc[0], singer_loss, singer_acc[0],  summary_str


	def read_input_file(self, file_name):
		"""
		Function to read and process input file, given name and the synth_mode.
		Returns features for the file based on mode (0 for hdf5 file, 1 for wav file).
		Currently, only the HDF5 version is implemented.
		"""
		# if file_name.endswith('.hdf5'):

		stat_file = h5py.File(config.stat_dir+'stats.hdf5', mode='r')
		max_voc = np.array(stat_file["voc_stft_maximus"])
		min_voc = np.array(stat_file["voc_stft_minimus"])

		max_feat = np.array(stat_file["feats_maximus"])
		min_feat = np.array(stat_file["feats_minimus"])

		stat_file.close()
		
		feat_file = h5py.File(config.voice_dir + file_name)

		voc_stft = np.array(feat_file['voc_stft'])[()]

		feats = np.array(feat_file['feats'])

		pho_target = np.array(feat_file["phonemes"])

		f0 = feats[:,-2]

		med = np.median(f0[f0 > 0])

		f0[f0==0] = med

		f0_nor = (f0 - min_feat[-2])/(max_feat[-2]-min_feat[-2])


		f0_quant = np.rint(f0_nor*config.num_f0) + 1

		f0_quant = f0_quant * (1-feats[:,-1]) 

		feat_file.close()

		in_batches_stft, nchunks_in = utils.generate_overlapadd(voc_stft)
		
		in_batches_stft = (np.array(in_batches_stft) - min_voc)/(max_voc - min_voc)



		in_batches_f0, nchunks_in = utils.generate_overlapadd(np.expand_dims(f0_quant,-1))

		in_batches_f0 = np.squeeze(in_batches_f0)

		in_batches_pho, nchunks_in = utils.generate_overlapadd(np.expand_dims(pho_target,-1))

		in_batches_pho = np.squeeze(in_batches_pho)

		return in_batches_stft, in_batches_f0, in_batches_pho, nchunks_in

	def test_file(self, file_name):
		"""
		Function to extract multi pitch from file. Currently supports only HDF5 files.
		"""
		sess = tf.Session()
		self.load_model(sess, log_dir = config.log_dir)
		scores = self.process_file(file_name, sess)
		return scores

	def process_file(self, file_name, sess):

		in_batches_stft, in_batches_f0, in_batches_pho, nchunks_in = self.read_input_file(file_name)

		out_batches_stft = []
		out_batches_pho = []
		for in_batch_stft, in_batch_f0 in zip(in_batches_stft, in_batches_f0):
			feed_dict = {self.input_placeholder: in_batch_stft,self.f0_labels: in_batch_f0, self.is_train: False}
			out_stft, out_pho = sess.run([self.output, self.pho_logits], feed_dict=feed_dict)
			out_batches_stft.append(out_stft)
			out_batches_pho.append(out_pho)

		out_batches_stft = np.array(out_batches_stft)
		out_batches_pho = np.array(out_batches_pho)


		out_batches_stft = utils.overlapadd(out_batches_stft,nchunks_in)
		in_batches_stft = utils.overlapadd(in_batches_stft,nchunks_in)

		out_batches_pho = utils.overlapadd(out_batches_pho,nchunks_in)
		in_batches_pho = utils.overlapadd(np.expand_dims(in_batches_pho, -1),nchunks_in)		

		# import pdb;pdb.set_trace()
		plt.figure(1)
		
		ax1 = plt.subplot(211)

		plt.imshow(np.log(in_batches_stft.T),aspect='auto',origin='lower')

		ax1.set_title("Ground Truth STFT", fontsize=10)

		ax3 =plt.subplot(212, sharex = ax1, sharey = ax1)

		ax3.set_title("Output STFT", fontsize=10)

		plt.imshow(np.log(out_batches_stft.T),aspect='auto',origin='lower')

		plt.figure(2)
		
		ax1 = plt.subplot(211)

		plt.imshow(in_batches_pho,aspect='auto',origin='lower')

		ax1.set_title("Ground Truth Phonemes", fontsize=10)

		ax3 =plt.subplot(212, sharex = ax1, sharey = ax1)

		ax3.set_title("Output Phonemes", fontsize=10)

		plt.imshow(out_batches_pho,aspect='auto',origin='lower')

		plt.show()

	def model(self):
		"""
		The main model function, takes and returns tensors.
		Defined in modules.

		"""

		with tf.variable_scope('Singer_Model') as scope:
			self.singer_emb, self.singer_logits = modules.singer_network(self.input_placeholder, self.is_train)
			self.singer_classes = tf.argmax(self.singer_logits, axis=-1)
			self.singer_probs = tf.nn.softmax(self.singer_logits)

		with tf.variable_scope('Phone_Model') as scope:
			self.pho_emb, self.pho_logits = modules.phone_network(self.input_placeholder, self.is_train)
			self.pho_classes = tf.argmax(self.pho_logits, axis=-1)
			self.pho_probs = tf.nn.softmax(self.pho_logits)


		with tf.variable_scope('F0_Model') as scope:
			self.f0_emb, self.f0_logits = modules.f0_network(self.input_placeholder, self.is_train)
			self.f0_classes = tf.argmax(self.f0_logits, axis=-1)
			self.f0_probs = tf.nn.softmax(self.f0_logits)

		with tf.variable_scope('Final_Model') as scope:
			self.output = modules.full_network(self.pho_emb, self.singer_emb, self.f0_emb, self.is_train)


def test():
	# model = DeepSal()
	# # model.test_file('nino_4424.hdf5')
	# model.test_wav_folder('./helena_test_set/', './results/')

	model = MultiSynth()
	model.train()

if __name__ == '__main__':
	test()





