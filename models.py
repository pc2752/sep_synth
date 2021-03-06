import tensorflow as tf
import modules_tf as modules
import config
from data_pipeline import data_gen_pho, data_gen_full, data_gen_med
import time, os
import utils
import h5py
import numpy as np
import mir_eval
import pandas as pd
from random import randint
import librosa
# import sig_process

import soundfile as sf

import matplotlib.pyplot as plt
from scipy.ndimage import filters


def one_hotize(inp, max_index=config.num_phos):


    output = np.eye(max_index)[inp.astype(int)]

    return output

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


        # self.pho_weights = tf.reduce_sum(config.phonemas_weights * self.phone_onehot_labels, axis=-1)

        # self.unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.phone_onehot_labels, logits = self.pho_logits)

        # self.weighted_losses = self.unweighted_losses * self.pho_weights

        self.pho_loss = self.pho_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.phone_onehot_labels, logits = self.pho_logits))

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

        self.input_placeholder = tf.placeholder(tf.float32, shape=(None, config.max_phr_len, config.num_phos),
                                           name='input_placeholder')

        self.cond_placeholder = tf.placeholder(tf.float32, shape=(None, config.max_phr_len,config.input_features),
                                           name='cond_placeholder')

        self.phone_onehot_labels = tf.placeholder(tf.float32, shape=(None, config.max_phr_len, config.num_phos),
                                           name='output_placeholder')       

        self.phoneme_labels = tf.argmax(self.phone_onehot_labels, axis=-1)

        self.is_train = tf.placeholder(tf.bool, name="is_train")





    def train(self):
        """
        Function to train the model, and save Tensorboard summary, for N epochs. 
        """
        sess = tf.Session()


        self.loss_function()
        self.get_optimizers()
        self.load_model(sess, config.log_dir_phone)
        self.get_summary(sess, config.log_dir_phone)
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

                    cond = np.roll(phons, 1, 1)

                    cond[:,0,:] = 0

                    cond = cond + np.random.normal(0,.5,(cond.shape)) *0.8 

                    step_loss, step_acc, summary_str = self.train_model(spec, phons, cond, sess)
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
                accuracy, file_name = self.validate_file(sess)
                print_dict["Validation Accuracy for {}".format(file_name)] = accuracy

            batch_num = 0
            with tf.variable_scope('Validation'):
                for spec, phons in val_generator:
                    cond = np.roll(phons, 1, 1)

                    cond[:,0,:] = 0

                    cond = cond + np.random.normal(0,.5,(cond.shape)) * 0.4 

                    step_loss, step_acc, summary_str = self.validate_model(spec, phons, cond, sess)
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

    def train_model(self, spec, phons, cond, sess):
        """
        Function to train the model for each epoch
        """
        feed_dict = {self.cond_placeholder: spec, self.phone_onehot_labels: phons, self.input_placeholder: cond, self.is_train: True}
        _, step_loss, step_acc = sess.run(
            [self.train_function, self.pho_loss, self.pho_acc], feed_dict=feed_dict)
        summary_str = sess.run(self.summary, feed_dict=feed_dict)

        return step_loss, step_acc[0], summary_str

    def validate_model(self, spec, phons, cond, sess):
        """
        Function to train the model for each epoch
        """
        feed_dict = {self.cond_placeholder: spec, self.phone_onehot_labels: phons, self.input_placeholder: cond, self.is_train: False}

        step_loss, step_acc = sess.run([self.pho_loss, self.pho_acc], feed_dict=feed_dict)
        summary_str = sess.run(self.summary, feed_dict=feed_dict)
        return step_loss, step_acc[0], summary_str

    def validate_file(self, sess):
        """
        Function to train the model for each epoch
        """
        val_list = [x for x in os.listdir(config.voice_dir) if x.endswith('.hdf5')and x.startswith('nus') and x.split('_')[1] in ['ADIZ', 'JLEE', 'JTAN', 'KENN'] and not x.startswith('nus_KENN_read')]
        voc_index = np.random.randint(0,len(val_list))
        voc_to_open = val_list[voc_index]

        voc_stft, pho_target = self.read_hdf5_file(voc_to_open)

        index = np.random.randint(0,len(voc_stft) - 2000)

        out_phonemes = self.process_file(voc_stft[index:index+2000], sess)

        accuracy = np.equal(pho_target[index:index+2000], np.argmax(out_phonemes, -1)).sum()/len(out_phonemes)

        return accuracy, voc_to_open

    def read_hdf5_file(self, file_name):
        """
        Function to read and process input file, given name and the synth_mode.
        Returns features for the file based on mode (0 for hdf5 file, 1 for wav file).
        Currently, only the HDF5 version is implemented.
        """
        # if file_name.endswith('.hdf5'):
        stat_file = h5py.File(config.stat_dir+'stats.hdf5', mode='r')

        max_feat = np.array(stat_file["feats_maximus"])
        min_feat = np.array(stat_file["feats_minimus"])
        max_voc = np.array(stat_file["voc_stft_maximus"])
        min_voc = np.array(stat_file["voc_stft_minimus"])
        max_back = np.array(stat_file["back_stft_maximus"])
        min_back = np.array(stat_file["back_stft_minimus"])
        stat_file.close()

        with h5py.File(config.voice_dir + file_name) as feat_file:

            pho_target = np.array(feat_file["phonemes"])
            voc_stft = np.array(feat_file['voc_stft'])



        # pho_target = one_hotize(pho_target)

        voc_stft = (np.array(voc_stft) - min_voc)/(max_voc - min_voc)



        return voc_stft, pho_target

    def process_file(self, condi, sess):

        conds = np.zeros((config.batch_size, config.max_phr_len, config.input_features))
        outs = []
        inps = np.zeros((config.batch_size, config.max_phr_len, config.num_phos))

        count = 0

        for con in condi:
            conds = np.roll(conds, -1, 1)
            conds[:,-1, :] = con
            feed_dict = {self.input_placeholder: inps ,self.cond_placeholder: conds, self.is_train: False}
            frame_op = sess.run(self.pho_probs, feed_dict=feed_dict)
            outs.append(frame_op[0,-1,:])
            inps = np.roll(inps, -1, 1)
            inps[:,-1,:] = frame_op[:,-1,:]
            count+=1
            utils.progress(count,len(condi), suffix = 'Prediction Done')
        return np.array(outs)


    def model(self):
        """
        The main model function, takes and returns tensors.
        Defined in modules.

        """
        with tf.variable_scope('phone_Model') as scope:
            # regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
            self.pho_logits = modules.wave_archi(self.input_placeholder, self.cond_placeholder, self.is_train)
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
        self.d_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'Discriminator')

        self.pho_optimizer = tf.train.AdamOptimizer(learning_rate = config.init_lr)
        self.final_optimizer = tf.train.RMSPropOptimizer(learning_rate=5e-5)
        self.dis_optimizer = tf.train.RMSPropOptimizer(learning_rate=5e-5)
        self.singer_optimizer = tf.train.AdamOptimizer(learning_rate = config.init_lr)
        self.f0_optimizer = tf.train.AdamOptimizer(learning_rate = config.init_lr)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.global_step_dis = tf.Variable(0, name='dis_global_step', trainable=False)
        self.global_step_pho = tf.Variable(0, name='pho_global_step', trainable=False)
        self.global_step_f0 = tf.Variable(0, name='f0_global_step', trainable=False)
        self.global_step_singer = tf.Variable(0, name='singer_global_step', trainable=False)


        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.final_train_function = self.final_optimizer.minimize(self.final_loss, global_step = self.global_step, var_list = self.final_params)
            self.dis_train_function = self.dis_optimizer.minimize(self.D_loss, global_step = self.global_step_dis, var_list = self.d_params)
            self.pho_train_function = self.pho_optimizer.minimize(self.pho_loss, global_step = self.global_step_pho, var_list = self.pho_params)
            self.f0_train_function = self.f0_optimizer.minimize(self.f0_loss, global_step = self.global_step_f0, var_list = self.f0_params)
            self.singer_train_function = self.singer_optimizer.minimize(self.singer_loss, global_step = self.global_step_singer, var_list = self.singer_params)
            self.clip_discriminator_var_op = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in self.d_params]

    def loss_function(self):
        """
        returns the loss function for the model, based on the mode. 
        """

        # self.pho_weights = tf.reduce_sum(config.phonemas_weights * self.phone_onehot_labels, axis=-1)

        # self.unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.phone_onehot_labels, logits = self.pho_logits)

        # self.weighted_losses = self.unweighted_losses * self.pho_weights

        self.pho_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.phone_onehot_labels, logits = self.pho_logits))

#       self.pho_acc = tf.metrics.accuracy(labels = self.phoneme_labels, predictions = self.pho_classes)
        self.pho_acc = tf.metrics.accuracy(labels = tf.argmax(self.phone_onehot_labels, axis = -1), predictions = self.pho_classes)

        self.pho_acc_val = tf.metrics.accuracy(labels = tf.argmax(self.phone_onehot_labels, axis = -1), predictions = self.pho_classes)

        self.singer_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.singer_onehot_labels, logits=self.singer_logits))


        # self.singer_acc = tf.metrics.accuracy(labels=self.singer_labels , predictions=self.singer_classes)

        self.singer_acc = tf.metrics.accuracy(labels = tf.argmax(self.singer_onehot_labels, axis = -1) , predictions=self.singer_classes)

        self.singer_acc_val = tf.metrics.accuracy(labels = tf.argmax(self.singer_onehot_labels, axis = -1) , predictions=self.singer_classes)


        self.f0_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.f0_onehot_labels, logits=self.f0_logits))

        # self.f0_acc = tf.metrics.accuracy(labels=self.f0_labels , predictions=self.f0_classes)

        self.final_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels= self.output_placeholder, logits = self.output)) 
        # tf.reduce_sum(tf.abs(self.input_placeholder- self.output))
        # 
        # + tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels= self.wave_placeholder, logits = self.output_wav))

        self.f0_acc = tf.metrics.accuracy(labels = tf.argmax(self.f0_onehot_labels, axis = -1) , predictions=self.f0_classes)

        self.f0_acc_val = tf.metrics.accuracy(labels = tf.argmax(self.f0_onehot_labels, axis = -1) , predictions=self.f0_classes)

        self.final_loss = tf.reduce_sum(tf.abs(self.output_placeholder- (self.output/2+0.5)))/(config.batch_size*config.max_phr_len*64) + tf.reduce_mean(self.D_fake+1e-12)

        self.D_loss = tf.reduce_mean(self.D_real +1e-12) - tf.reduce_mean(self.D_fake+1e-12)


        # + self.pho_loss + self.singer_loss + self.f0_loss
        # tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels= self.output_placeholder, logits = self.output)) 
        # tf.reduce_sum(tf.abs(self.input_placeholder- self.output))
        # 
        # + tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels= self.wave_placeholder, logits = self.output_wav))


    def get_summary(self, sess, log_dir):
        """
        Gets the summaries and summary writers for the losses.
        """

        self.pho_summary = tf.summary.scalar('pho_loss', self.pho_loss)

        self.pho_acc_summary = tf.summary.scalar('pho_accuracy', self.pho_acc[0])

        self.pho_acc_summary_val = tf.summary.scalar('pho_accuracy', self.pho_acc_val[0])

        self.final_summary = tf.summary.scalar('final_loss', self.final_loss)

        self.f0_summary = tf.summary.scalar('f0_loss', self.f0_loss)

        self.f0_acc_summary = tf.summary.scalar('f0_accuracy', self.f0_acc[0])

        self.f0_acc_summary_val = tf.summary.scalar('f0_accuracy', self.f0_acc_val[0])

        self.singer_summary = tf.summary.scalar('singer_loss', self.singer_loss)

        self.singer_acc_summary = tf.summary.scalar('singer_accuracy', self.singer_acc[0])

        self.singer_acc_summary_val = tf.summary.scalar('singer_accuracy', self.singer_acc_val[0])



        self.train_summary_writer = tf.summary.FileWriter(log_dir+'train/', sess.graph)
        self.val_summary_writer = tf.summary.FileWriter(log_dir+'val/', sess.graph)
        self.summary = tf.summary.merge([self.pho_summary,self.final_summary, self.pho_acc_summary,self.f0_summary, self.f0_acc_summary, self.singer_summary,self.singer_acc_summary])
        self.summary_val = tf.summary.merge([self.pho_summary,self.final_summary, self.pho_acc_summary_val,self.f0_summary, self.f0_acc_summary_val, self.singer_summary,self.singer_acc_summary_val])

    def get_placeholders(self):
        """
        Returns the placeholders for the model. 
        Depending on the mode, can return placeholders for either just the generator or both the generator and discriminator.
        """

        self.input_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size, config.max_phr_len, config.input_features),
                                           name='input_placeholder')

        self.input_placeholder_singer = tf.placeholder(tf.float32, shape=(config.batch_size, config.max_phr_len, config.input_features),
                                           name='input_placeholder_singer')

        self.output_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size, config.max_phr_len, config.output_features),
                                           name='output_placeholder')       


        # self.phoneme_labels = tf.placeholder(tf.int32, shape=(config.batch_size, config.max_phr_len),
        #                                 name='phoneme_placeholder')
        # self.phone_onehot_labels = tf.one_hot(indices=tf.cast(self.phoneme_labels, tf.int32), depth = config.num_phos)
        self.phone_onehot_labels = tf.placeholder(tf.float32, shape=(config.batch_size, config.max_phr_len, config.num_phos),
                                        name='phoneme_placeholder')

        self.f0_onehot_labels = tf.placeholder(tf.float32, shape=(config.batch_size, config.max_phr_len, config.num_f0),
                                        name='f0_placeholder')
        
        # self.f0_labels = tf.placeholder(tf.int32, shape=(config.batch_size, config.max_phr_len),
        #                                 name='f0_placeholder')
        # self.f0_onehot_labels = tf.one_hot(indices=tf.cast(self.f0_labels, tf.int32), depth = config.num_f0)

        self.singer_onehot_labels = tf.placeholder(tf.float32, shape=(config.batch_size, config.num_singers),
            name='singer_placeholder')

        # self.singer_labels = tf.placeholder(tf.float32, shape=(config.batch_size),name='singer_placeholder')
        # self.singer_onehot_labels = tf.one_hot(indices=tf.cast(self.singer_labels, tf.int32), depth = config.num_singers)


        self.is_train = tf.placeholder(tf.bool, name="is_train")

        # self.wave_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size, config.max_phr_len*2**8),
        #                                  name='wave_placeholder')

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
            if epoch<500:
                data_generator = data_gen_full()
                val_generator = data_gen_full(mode = 'Val')
                epoch_pho_loss = 0
                epoch_pho_acc = 0

                epoch_f0_loss = 0
                epoch_f0_acc = 0

                epoch_singer_loss = 0
                epoch_singer_acc = 0
                val_final_loss = 0
                val_dis_loss = 0


                val_pho_loss = 0
                val_pho_acc = 0

                val_f0_loss = 0
                val_f0_acc = 0
            else:
                data_generator = data_gen_med()
                val_generator = data_gen_med(mode = 'Val')

            start_time = time.time()

            batch_num = 0
            epoch_final_loss = 0
            epoch_dis_loss = 0






            val_singer_loss = 0
            val_singer_acc = 0

            with tf.variable_scope('Training'):
                for mix_in, singer_targs, voc_out, f0_out, pho_targs in data_generator:

                    if epoch< 500:

                        final_loss, dis_loss, singer_loss, singer_acc, f0_loss, f0_acc, pho_loss, pho_acc, summary_str = self.train_model(mix_in, singer_targs, voc_out, f0_out,pho_targs, epoch, sess)
                        epoch_pho_loss+=pho_loss
                        epoch_pho_acc+=pho_acc

                        epoch_f0_loss+=f0_loss
                        epoch_f0_acc+=f0_acc

                        epoch_singer_loss+=singer_loss
                        epoch_singer_acc+=singer_acc 
                    else:
                        final_loss, dis_loss, summary_str = self.train_model_2(mix_in, voc_out, sess)



                    epoch_final_loss+=final_loss
                    epoch_dis_loss+=dis_loss




                    self.train_summary_writer.add_summary(summary_str, epoch)
                    self.train_summary_writer.flush()

                    utils.progress(batch_num,config.batches_per_epoch_train, suffix = 'training done')

                    batch_num+=1

                epoch_final_loss = epoch_final_loss/batch_num
                epoch_dis_loss = epoch_dis_loss/batch_num

                if epoch<500:

                    epoch_singer_loss = epoch_singer_loss/batch_num
                    epoch_singer_acc = epoch_singer_acc/batch_num

                    epoch_pho_loss = epoch_pho_loss/batch_num
                    epoch_pho_acc = epoch_pho_acc/batch_num

                    epoch_f0_loss = epoch_f0_loss/batch_num
                    epoch_f0_acc = epoch_f0_acc/batch_num

                print_dict = {"Final Loss": epoch_final_loss}

                print_dict["Dis Loss"] =  epoch_dis_loss

                # print_dict["Pho Loss"] =  epoch_pho_loss
                print_dict["Pho Accuracy"] =  epoch_pho_acc

                # print_dict["F0 Loss"] =  epoch_f0_loss
                print_dict["F0 Accuracy"] =  epoch_f0_acc

                # print_dict["Singer Loss"] =  epoch_singer_loss
                print_dict["Singer Accuracy"] =  epoch_singer_acc


            if (epoch + 1) % config.validate_every == 0:
                batch_num = 0
                with tf.variable_scope('Validation'):
                    for mix_in, singer_targs, voc_out, f0_out, pho_targs in val_generator:

                        if epoch< 500:    
                            final_loss, dis_loss, singer_loss, singer_acc, f0_loss, f0_acc, pho_loss, pho_acc, summary_str= self.validate_model(mix_in, singer_targs, voc_out, f0_out,pho_targs, sess)

                            val_pho_loss+=pho_loss
                            val_pho_acc+=pho_acc

                            val_f0_loss+=f0_loss
                            val_f0_acc+=f0_acc

                            val_singer_loss+=singer_loss
                            val_singer_acc+=singer_acc  

                        else:
                            final_loss, dis_loss, summary_str = self.validate_model_2(mix_in, voc_out, sess)

                        val_final_loss+=final_loss
                        val_dis_loss+=dis_loss

                        self.val_summary_writer.add_summary(summary_str, epoch)
                        self.val_summary_writer.flush()
                        batch_num+=1

                        utils.progress(batch_num, config.batches_per_epoch_val, suffix='validation done')

                    val_final_loss = val_final_loss/batch_num
                    val_dis_loss = val_dis_loss/batch_num
                    if epoch<500:

                        val_pho_loss = val_pho_loss/batch_num
                        val_pho_acc = val_pho_acc/batch_num

                        val_f0_loss = val_f0_loss/batch_num
                        val_f0_acc = val_f0_acc/batch_num

                        val_singer_loss = val_singer_loss/batch_num
                        val_singer_acc = val_singer_acc/batch_num


                    print_dict["Val Final Loss"] =  val_final_loss
                    print_dict["Val Dis Loss"] =  val_dis_loss

                    # print_dict["Val Pho Loss"] =  val_pho_loss
                    print_dict["Val Pho Accuracy"] =  val_pho_acc

                    # print_dict["Val F0 Loss"] =  val_f0_loss
                    print_dict["Val F0 Accuracy"] =  val_f0_acc

                    # print_dict["Val Singer Loss"] =  val_singer_loss
                    print_dict["Val Singer Accuracy"] =  val_singer_acc


            end_time = time.time()
            if (epoch + 1) % config.print_every == 0:
                self.print_summary(print_dict, epoch, end_time-start_time)
            if (epoch + 1) % config.save_every == 0 or (epoch + 1) == config.num_epochs:
                self.save_model(sess, epoch+1, config.log_dir)


    def train_model(self, mix_in, singer_targs, voc_out, f0_out, pho_targs, epoch, sess):
        """
        Function to train the model for each epoch
        """
        # assert (np.argmax(singer_targs, axis = -1)>3).all()

        # mix_in = np.clip(mix_in + np.random.normal(0,.5,(mix_in.shape)) * 0.4 ,0.0, 1.0)

        if epoch<25 or epoch%100 == 0:
            n_critic = 15
        else:
            n_critic = 5
        feed_dict = {self.input_placeholder: mix_in,self.input_placeholder_singer: mix_in, self.singer_onehot_labels: singer_targs, self.output_placeholder: voc_out, self.f0_onehot_labels: f0_out, self.phone_onehot_labels: pho_targs, self.is_train: True}
        for critic_itr in range(n_critic):
            sess.run(self.dis_train_function, feed_dict = feed_dict)
            sess.run(self.clip_discriminator_var_op, feed_dict = feed_dict)

        teacher_train = np.random.rand(1)<0.5

        if epoch<100 or not teacher_train:
            
            _,_, _,_, final_loss, dis_loss,  singer_loss, singer_acc, pho_loss, pho_acc, f0_loss, f0_acc = sess.run(
                [self.final_train_function, self.pho_train_function, self.f0_train_function, self.singer_train_function, self.final_loss, self.D_loss, self.singer_loss, self.singer_acc, self.pho_loss, self.pho_acc, self.f0_loss, self.f0_acc], feed_dict=feed_dict)
        else:
            feed_dict = {self.input_placeholder: mix_in,self.input_placeholder_singer: mix_in, self.output_placeholder: voc_out, self.is_train: False}
            f0_est, pho_est, singer_est = sess.run([self.f0_probs, self.pho_probs, self.singer_probs], feed_dict=feed_dict)
            feed_dict = {self.input_placeholder: mix_in,self.input_placeholder_singer: mix_in, self.singer_onehot_labels: singer_est, self.output_placeholder: voc_out, self.f0_onehot_labels: f0_est, self.phone_onehot_labels: pho_est, self.is_train: True}
            _,_, _,_, final_loss, dis_loss,  singer_loss, singer_acc, pho_loss, pho_acc, f0_loss, f0_acc = sess.run(
                [self.final_train_function, self.pho_train_function, self.f0_train_function, self.singer_train_function, self.final_loss, self.D_loss, self.singer_loss, self.singer_acc, self.pho_loss, self.pho_acc, self.f0_loss, self.f0_acc], feed_dict=feed_dict)

        summary_str = sess.run(self.summary, feed_dict=feed_dict)


        return final_loss, dis_loss, singer_loss, singer_acc[0], f0_loss, f0_acc[0], pho_loss, pho_acc[0], summary_str
 
    def train_model_2(self, mix_in, voc_out, sess):
        """
        Function to train the model for each epoch
        """
        # assert (np.argmax(singer_targs, axis = -1)>3).all()

        # mix_in = np.clip(mix_in + np.random.normal(0,.5,(mix_in.shape)) * 0.4 ,0.0, 1.0)

        feed_dict = {self.input_placeholder: mix_in,self.input_placeholder_singer: mix_in, self.output_placeholder: voc_out, self.is_train: False}
        f0_est, pho_est, singer_est = sess.run([self.f0_probs, self.pho_probs, self.singer_probs], feed_dict=feed_dict)
        feed_dict = {self.input_placeholder: mix_in,self.input_placeholder_singer: mix_in, self.singer_onehot_labels: singer_est, self.output_placeholder: voc_out, self.f0_onehot_labels: f0_est, self.phone_onehot_labels: pho_est, self.is_train: True}
        _,_, _,_, final_loss, dis_loss,  singer_loss, singer_acc, pho_loss, pho_acc, f0_loss, f0_acc = sess.run(
            [self.final_train_function, self.pho_train_function, self.f0_train_function, self.singer_train_function, self.final_loss, self.D_loss, self.singer_loss, self.singer_acc, self.pho_loss, self.pho_acc, self.f0_loss, self.f0_acc], feed_dict=feed_dict)

        summary_str = sess.run(self.summary, feed_dict=feed_dict)


        return final_loss, dis_loss, summary_str

    def validate_model(self,mix_in, singer_targs, voc_out, f0_out,pho_targs, sess):
        """
        Function to train the model for each epoch
        """
        # assert (np.argmax(singer_targs, axis = -1)<4).all()
        feed_dict = {self.input_placeholder: mix_in,self.input_placeholder_singer: mix_in, self.singer_onehot_labels: singer_targs, self.output_placeholder: voc_out, self.f0_onehot_labels: f0_out, self.phone_onehot_labels: pho_targs, self.is_train: False}

        final_loss, dis_loss, singer_loss, singer_acc, pho_loss, pho_acc, f0_loss, f0_acc = sess.run(
            [self.final_loss,self.D_loss, self.singer_loss, self.singer_acc_val, self.pho_loss, self.pho_acc_val, self.f0_loss, self.f0_acc_val], feed_dict=feed_dict)

        summary_str = sess.run(self.summary_val, feed_dict=feed_dict)


        return final_loss, dis_loss, singer_loss, singer_acc[0], f0_loss, f0_acc[0], pho_loss, pho_acc[0], summary_str

    def validate_model_2(self, mix_in, voc_out, sess):
        """
        Function to train the model for each epoch
        """
        # assert (np.argmax(singer_targs, axis = -1)>3).all()

        # mix_in = np.clip(mix_in + np.random.normal(0,.5,(mix_in.shape)) * 0.4 ,0.0, 1.0)

        feed_dict = {self.input_placeholder: mix_in,self.input_placeholder_singer: mix_in, self.output_placeholder: voc_out, self.is_train: False}


        f0_est, pho_est, singer_est = sess.run([self.f0_probs, self.pho_probs, self.singer_probs], feed_dict=feed_dict)
        feed_dict = {self.input_placeholder: mix_in,self.input_placeholder_singer: mix_in, self.singer_onehot_labels: singer_est, self.output_placeholder: voc_out, self.f0_onehot_labels: f0_est, self.phone_onehot_labels: pho_est, self.is_train: False}
        _,_, _,_, final_loss, dis_loss,  singer_loss, singer_acc, pho_loss, pho_acc, f0_loss, f0_acc = sess.run(
            [self.final_train_function, self.pho_train_function, self.f0_train_function, self.singer_train_function, self.final_loss, self.D_loss, self.singer_loss, self.singer_acc, self.pho_loss, self.pho_acc, self.f0_loss, self.f0_acc], feed_dict=feed_dict)

        summary_str = sess.run(self.summary, feed_dict=feed_dict)


        return final_loss, dis_loss, summary_str

    def read_hdf5_file(self, file_name):
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

        # pho_target = np.array(feat_file["phonemes"])

        f0 = feats[:,-2]

        med = np.median(f0[f0 > 0])

        f0[f0==0] = med

        f0_nor = (f0 - min_feat[-2])/(max_feat[-2]-min_feat[-2])


        f0_quant = np.rint(f0_nor*config.num_f0) + 1

        f0_quant = f0_quant * (1-feats[:,-1]) 

        feat_file.close()

        voc_stft = np.clip(voc_stft, 0.0, 1.0)

        return voc_stft, feats

    def read_wav_file(self, file_name):

        audio, fs = librosa.core.load(file_name, sr=config.fs)

        audio = np.float64(audio)

        if len(audio.shape) == 2:

            vocals = np.array((audio[:,1]+audio[:,0])/2)

        else: 
            vocals = np.array(audio)

        voc_stft = abs(utils.stft(vocals))

        feats = utils.stft_to_feats(vocals,fs)

        voc_stft = np.clip(voc_stft, 0.0, 1.0)

        return voc_stft, feats



    def test_file_wav(self, file_name, file_name_singer):
        """
        Function to extract multi pitch from file. Currently supports only HDF5 files.
        """
        sess = tf.Session()
        self.load_model(sess, log_dir =  config.log_dir)
        voc_stft, feats = self.read_wav_file(file_name)

        voc_stft_singer, feats_singer = self.read_hdf5_file(file_name_singer)
        out_feats = self.process_file(voc_stft, voc_stft_singer, sess)

        self.plot_features(feats, out_feats)

        import pdb;pdb.set_trace()

        out_featss = np.concatenate((out_feats[:feats.shape[0]], feats[:out_feats.shape[0],-2:]), axis = -1)

        utils.feats_to_audio(out_featss[:5000],file_name[:-4]+'gan_op.wav') 


    def test_file_hdf5(self, file_name, file_name_singer):
        """
        Function to extract multi pitch from file. Currently supports only HDF5 files.
        """
        sess = tf.Session()
        self.load_model(sess, log_dir = config.log_dir)
        voc_stft, feats = self.read_hdf5_file(file_name)

        voc_stft_singer, feats_singer = self.read_hdf5_file(file_name_singer)
        out_feats = self.process_file(voc_stft, voc_stft_singer, sess)

        self.plot_features(feats, out_feats)

        import pdb;pdb.set_trace()

        out_featss = np.concatenate((out_feats[:feats.shape[0]], feats[:out_feats.shape[0],-2:]), axis = -1)

        utils.feats_to_audio(out_featss,file_name[:-4]+'gan_op.wav') 

    def plot_features(self, feats, out_feats):

        plt.figure(1)
        
        ax1 = plt.subplot(211)

        plt.imshow(feats[:,:-2].T,aspect='auto',origin='lower')

        ax1.set_title("Ground Truth STFT", fontsize=10)

        ax3 =plt.subplot(212, sharex = ax1, sharey = ax1)

        ax3.set_title("Output STFT", fontsize=10)

        plt.imshow(out_feats[:,:-1].T,aspect='auto',origin='lower')

        plt.figure(2)

        plt.plot(feats[:,-2])
        plt.plot(out_feats[:,-1])


        plt.show()


    def process_file(self, voc_stft, voc_stft_singer, sess):

        stat_file = h5py.File(config.stat_dir+'stats.hdf5', mode='r')

        max_voc = np.array(stat_file["voc_stft_maximus"])
        min_voc = np.array(stat_file["voc_stft_minimus"])

        max_feat = np.array(stat_file["feats_maximus"])
        min_feat = np.array(stat_file["feats_minimus"])
        stat_file.close()


        if len(voc_stft)>len(voc_stft_singer):
            voc_stft = voc_stft[:len(voc_stft_singer)]
        else:
            voc_stft_singer = voc_stft_singer[:len(voc_stft)]

        in_batches_stft, nchunks_in = utils.generate_overlapadd(voc_stft)

        in_batches_stft_singer, nchunks_in_singer = utils.generate_overlapadd(voc_stft_singer)

        in_batches_stft = np.clip(in_batches_stft, 0.0, 1.0) 

        in_batches_stft_singer = np.clip(in_batches_stft_singer, 0.0, 1.0) 


        out_batches_feats = []
        out_batches_f0 = []

        out_batches_singer = []

        # for in_batch_stft_singer in in_batches_stft_singer:
        #     feed_dict = {self.input_placeholder_singer: in_batch_stft_singer, self.is_train: False}
        #     singer_est = sess.run(self.singer_probs, feed_dict=feed_dict)
        #     out_batches_singer.append(singer_est)

        # singer_emb = np.tile(np.mean(np.mean(np.array(out_batches_singer), axis = 0), axis = 0), [config.batch_size, 1])


        # singer_emb = np.tile(one_hotize(np.argmax(np.mean(np.mean(np.array(out_batches_singer), axis = 0), axis = 0), axis = -1), config.num_singers), [config.batch_size, 1])
        # pho_est = one_hotize(np.argmax(pho_est, axis = -1), config.num_phos)

        for in_batch_stft, in_batch_stft_singer in zip(in_batches_stft, in_batches_stft_singer) :
            feed_dict = {self.input_placeholder: in_batch_stft, self.is_train: False}
            f0_est, pho_est = sess.run([self.f0_probs, self.pho_probs], feed_dict=feed_dict)
            feed_dict = {self.input_placeholder: in_batch_stft, self.input_placeholder_singer: in_batch_stft_singer, self.f0_onehot_labels: f0_est, self.phone_onehot_labels: pho_est, self.is_train: False}
            out_feats = sess.run(self.output, feed_dict=feed_dict)
            out_batches_feats.append(out_feats)
            out_batches_f0.append(f0_est)

        out_batches_feats = np.array(out_batches_feats)

        out_batches_feats = utils.overlapadd(out_batches_feats,nchunks_in)

        # out_batches_wav = utils.overlapadd(out_batches_wav,nchunks_in)

        # out_batches_wav = utils.overlapadd(np.expand_dims(out_batches_wav, -1),nchunks_in, overlap = config.max_phr_len*2**7) 

        # out_batches_wav = out_batches_wav *2 -1

        out_batches_feats = out_batches_feats*(max_feat[:-1] - min_feat[:-1]) + min_feat[:-1]

        return out_batches_feats



    def model(self):
        """
        The main model function, takes and returns tensors.
        Defined in modules.

        """

        with tf.variable_scope('Singer_Model') as scope:
            self.singer_emb, self.singer_logits = modules.singer_network(self.input_placeholder_singer, self.is_train)
            self.singer_classes = tf.argmax(self.singer_logits, axis=-1)
            self.singer_probs = tf.nn.softmax(self.singer_logits)

        with tf.variable_scope('Phone_Model') as scope:
            self.pho_logits = modules.phone_network(self.input_placeholder, self.is_train)
            self.pho_classes = tf.argmax(self.pho_logits, axis=-1)
            self.pho_probs = tf.nn.softmax(self.pho_logits)

        with tf.variable_scope('F0_Model') as scope:
            self.f0_logits = modules.f0_network(self.input_placeholder, self.is_train)
            self.f0_classes = tf.argmax(self.f0_logits, axis=-1)
            self.f0_probs = tf.nn.softmax(self.f0_logits)

        with tf.variable_scope('Final_Model') as scope:
            self.output = modules.full_network(self.input_placeholder, self.phone_onehot_labels, self.f0_onehot_labels, self.singer_emb, self.is_train)
            # self.output_decoded = tf.nn.sigmoid(self.output)
            # self.output_wav_decoded = tf.nn.sigmoid(self.output_wav)
        with tf.variable_scope('Discriminator') as scope: 
            self.D_real = modules.discriminator((self.output_placeholder-0.5)*2, self.phone_onehot_labels, self.f0_onehot_labels, self.singer_onehot_labels, self.is_train)
            scope.reuse_variables()
            self.D_fake = modules.discriminator(self.output, self.phone_onehot_labels, self.f0_onehot_labels, self.singer_onehot_labels, self.is_train)



def test():
    # model = DeepSal()
    # # model.test_file('nino_4424.hdf5')
    # model.test_wav_folder('./helena_test_set/', './results/')

    model = MultiSynth()
    model.train()

if __name__ == '__main__':
    test()





