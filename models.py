import tensorflow as tf
# from modules_tf import DeepSalience, nr_wavenet
import config
# from data_pipeline import data_gen, sep_gen
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

    def get_placeholders(self):
        """
        Returns the placeholders for the model. 
        Depending on the mode, can return placeholders for either just the generator or both the generator and discriminator.
        """

        self.input_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size, config.max_phr_len, 66),
                                           name='input_placeholder')

        self.output_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size, config.max_phr_len, 64),
                                            name='output_placeholder')

        self.f0_input_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size, config.max_phr_len, 1),
                                              name='f0_input_placeholder')

        self.rand_input_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size, config.max_phr_len, 4),
                                                name='rand_input_placeholder')

        # pho_input_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size,config.max_phr_len, 42),name='pho_input_placeholder')

        self.prob = tf.placeholder_with_default(1.0, shape=())

        self.phoneme_labels = tf.placeholder(tf.int32, shape=(config.batch_size, config.max_phr_len),
                                        name='phoneme_placeholder')
        self.phone_onehot_labels = tf.one_hot(indices=tf.cast(phoneme_labels, tf.int32), depth=42)

        self.singer_labels = tf.placeholder(tf.float32, shape=(config.batch_size), name='singer_placeholder')
        self.singer_onehot_labels = tf.one_hot(indices=tf.cast(singer_labels, tf.int32), depth=12)

        self.is_train = tf.placeholder(tf.bool, name="is_train")

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


    def get_optimizers(self):
        """
        Returns the optimizers for the model, based on the loss functions and the mode. 
        """
        self.optimizer = tf.train.AdamOptimizer(learning_rate = config.init_lr)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_function = self.optimizer.minimize(self.loss, global_step = self.global_step)


    def get_summary(self, sess, log_dir):
        """
        Gets the summaries and summary writers for the losses.
        """

        self.pho_summary = tf.summary.scalar('pho_loss', self.pho_loss)

        self.pho_acc_summary = tf.summary.scalar('pho_accuracy', self.pho_acc[0])


        self.train_summary_writer = tf.summary.FileWriter(log_dir+'train/', sess.graph)
        self.val_summary_writer = tf.summary.FileWriter(log_dir+'val/', sess.graph)
        self.summary = tf.summary.merge_all()
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


class MultiSynth(Model):

    def loss_function(self):
        """
        returns the loss function for the model, based on the mode. 
        """
        self.final_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="Final_Model")

        self.g_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="Generator")

        self.d_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="Discriminator")

        self.phone_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="phone_Model")



        # Phoneme network loss and summary

        self.pho_weights = tf.reduce_sum(config.phonemas_weights * phone_onehot_labels, axis=-1)

        self.unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=phone_onehot_labels, logits=pho_logits)

        self.weighted_losses = unweighted_losses * pho_weights

        self.pho_loss = tf.reduce_mean(weighted_losses)

        self.pho_acc = tf.metrics.accuracy(labels=phoneme_labels, predictions=pho_classes)




        # Discriminator Loss


        # D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(D_real) , logits=D_real+1e-12))
        # D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(D_fake) , logits=D_fake+1e-12)) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(D_fake_2) , logits=D_fake_2+1e-12)) *0.5
        # D_loss_fake_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(D_fake_real) , logits=D_fake_real+1e-12))

        # D_loss_real = tf.reduce_mean(D_real+1e-12)
        # D_loss_fake = - tf.reduce_mean(D_fake+1e-12)
        # D_loss_fake_real = - tf.reduce_mean(D_fake_real+1e-12)


        # gradients = tf.gradients(d_hat, x_hat)[0] + 1e-6
        # slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        # gradient_penalty = tf.reduce_mean((slopes-1.0)**2)
        # errD += gradient_penalty
        # D_loss_fake_real = - tf.reduce_mean(D_fake_real)


        D_correct_pred = tf.equal(tf.round(tf.sigmoid(D_real)), tf.ones_like(D_real))

        D_correct_pred_fake = tf.equal(tf.round(tf.sigmoid(D_fake_real)), tf.ones_like(D_fake_real))

        D_accuracy = tf.reduce_mean(tf.cast(D_correct_pred, tf.float32))

        D_accuracy_fake = tf.reduce_mean(tf.cast(D_correct_pred_fake, tf.float32))



        D_loss = tf.reduce_mean(D_real +1e-12)-tf.reduce_mean(D_fake+1e-12)
        # -tf.reduce_mean(D_fake_real+1e-12)*0.001

        dis_summary = tf.summary.scalar('dis_loss', D_loss)

        dis_acc_summary = tf.summary.scalar('dis_acc', D_accuracy)

        dis_acc_fake_summary = tf.summary.scalar('dis_acc_fake', D_accuracy_fake)

        #Final net loss

        # G_loss_GAN = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels= tf.ones_like(D_real), logits=D_fake+1e-12)) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels= tf.ones_like(D_fake_2), logits=D_fake_2+1e-12))
        # + tf.reduce_sum(tf.abs(output_placeholder- (voc_output_2/2+0.5))*(1-input_placeholder[:,:,-1:])) *0.00001

        G_loss_GAN = tf.reduce_mean(D_fake+1e-12) + tf.reduce_sum(tf.abs(output_placeholder- (voc_output_2/2+0.5))) *0.00005
                     # + tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels= output_placeholder, logits=voc_output)) *0.000005
        #

        G_correct_pred = tf.equal(tf.round(tf.sigmoid(D_fake)), tf.ones_like(D_real))

        # G_correct_pred_2 = tf.equal(tf.round(tf.sigmoid(D_fake_2)), tf.ones_like(D_real))

        G_accuracy = tf.reduce_mean(tf.cast(G_correct_pred, tf.float32))

        gen_summary = tf.summary.scalar('gen_loss', G_loss_GAN)

        gen_acc_summary = tf.summary.scalar('gen_acc', G_accuracy)

        final_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels= output_placeholder, logits=voc_output)) \
                           # +tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels= output_placeholder, logits=voc_output_3))*0.5

        # reconstruct_loss = tf.reduce_sum(tf.abs(output_placeholder- (voc_output_2/2+0.5)))
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


    def test_wav_folder(self, folder_name, save_path):
        """
        Function to extract multi pitch from wav files in a folder
        """

        songs = next(os.walk(folder_name))[1]

        sess = tf.Session()
        self.load_model(sess, log_dir = config.log_dir)
        

        for song in songs:
        	count = 0
        	print ("Processing song %s" % song)
	        file_list = [x for x in os.listdir(os.path.join(folder_name, song)) if x.endswith('.wav') and not x.startswith('.')]
	        for file_name in file_list:
		        in_batches_hcqt, nchunks_in, max_len = self.read_input_wav_file(os.path.join(folder_name, song, file_name))
		        out_batches_atb = []
		        for in_batch_hcqt in in_batches_hcqt:
		            feed_dict = {self.input_placeholder: in_batch_hcqt, self.is_train: False}
		            out_atb = sess.run(self.outputs, feed_dict=feed_dict)
		            out_batches_atb.append(out_atb)
		        out_batches_atb = np.array(out_batches_atb)
		        out_batches_atb = utils.overlapadd(out_batches_atb.reshape(out_batches_atb.shape[0], config.batch_size, config.max_phr_len, -1),
		                         nchunks_in)
		        out_batches_atb = out_batches_atb[:max_len]
		        time_1, ori_freq = utils.process_output(out_batches_atb)
		        utils.save_multif0_output(time_1, ori_freq, os.path.join(save_path,song,file_name[:-4]+'.csv'))
		        count+=1
		        utils.progress(count, len(file_list), suffix='evaluation done')

    def extract_f0_file(self, file_name, sess):
        if file_name in config.val_list:
            mode = "Val"
        else:
            mode = "Train"
        num_singers = file_name.count('1')
        song_name = file_name.split('_')[0].capitalize()
        voice = config.log_dir.split('_')[-1][:-1].capitalize()

        in_batches_hcqt, atb, nchunks_in = self.read_input_file(file_name)
        out_batches_atb = []
        for in_batch_hcqt in in_batches_hcqt:
            feed_dict = {self.input_placeholder: in_batch_hcqt, self.is_train: False}
            out_atb = sess.run(self.outputs, feed_dict=feed_dict)
            out_batches_atb.append(out_atb)
        out_batches_atb = np.array(out_batches_atb)
        out_batches_atb = utils.overlapadd(out_batches_atb.reshape(out_batches_atb.shape[0], config.batch_size, config.max_phr_len, -1),
                         nchunks_in)
        out_batches_atb = out_batches_atb[:atb.shape[0]]

        baba = np.mean(np.equal(np.round(atb[atb>0]), np.round(out_batches_atb[atb>0])))

        atb = filters.gaussian_filter1d(atb.T, 0.5, axis=0, mode='constant').T


        plt.figure(1)
        plt.suptitle("Note Probabilities for song {}, voice {}, with {} singers, from the {} set".format(song_name, voice,num_singers, mode) + "bin activation accuracy: {0:.0%}".format(baba), fontsize=10)
        ax1 = plt.subplot(211)

        plt.imshow(np.round(atb.T), origin = 'lower', aspect = 'auto')

        ax1.set_title("Ground Truth Note Probabilities (10 cents per bin)", fontsize=10)
        ax2 = plt.subplot(212, sharex = ax1, sharey=ax1)
        plt.imshow(np.round(out_batches_atb.T), origin='lower', aspect='auto')
        ax2.set_title("Output Note Probabilities (10 cents per bin)", fontsize=10)
        plt.show()

        cont = utils.query_yes_no("Do you want to see probability distribution per frame? Default No", default = "no")

        while cont:

            num_sings = int(input("How many distinct pitches per frame to plot. Default {}".format(num_singers)) or num_singers)


            index = np.random.choice(np.where(atb.sum(axis=1)==num_sings)[0])
            plt.figure(1)
            plt.suptitle("Probability Distribution For one of the Frames With {} Distinct Pitches in GT".format(num_singers))
            ax1 = plt.subplot(211)
            ax1.set_title("Ground Truth Probability Distribution Across Frame", fontsize=10)
            plt.plot(np.round(atb[index]))
            ax2 = plt.subplot(212, sharex = ax1, sharey = ax1)
            plt.plot(np.round(out_batches_atb[index]))
            ax2.set_title("Output Probability Distribution Across Frame", fontsize=10)
            plt.show()
            cont = utils.query_yes_no("Do you want to see probability distribution per frame? Default No", default="no")

        #
        time_1, ori_freq = utils.process_output(atb)
        time_2, est_freq = utils.process_output(out_batches_atb)

        utils.save_multif0_output(time_1, ori_freq, './gt.csv')
        utils.save_multif0_output(time_2, est_freq, './op.csv')

        scores = mir_eval.multipitch.evaluate(time_1, ori_freq, time_2, est_freq)
        return scores

        # import pdb;pdb.set_trace()

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
            data_generator = data_gen()
            val_generator = data_gen(mode = 'Val')
            start_time = time.time()


            batch_num = 0
            epoch_train_loss = 0
            epoch_train_acc = 0
            epoch_val_loss = 0
            epoch_val_acc = 0

            with tf.variable_scope('Training'):
                for ins, outs in data_generator:

                    step_loss, step_acc, summary_str = self.train_model(ins, outs, sess)
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
                    for ins, outs in val_generator:
                        step_loss, step_acc, summary_str = self.validate_model(ins, outs, sess)
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

    def train_model(self, ins, outs, sess):
        """
        Function to train the model for each epoch
        """
        feed_dict = {self.input_placeholder: ins, self.output_placeholder: outs, self.is_train: True}
        _, step_loss, step_acc = sess.run(
            [self.train_function, self.loss, self.accuracy], feed_dict=feed_dict)
        summary_str = sess.run(self.summary, feed_dict=feed_dict)

        return step_loss, step_acc, summary_str

    def validate_model(self,ins, outs, sess):
        """
        Function to train the model for each epoch
        """
        feed_dict = {self.input_placeholder: ins, self.output_placeholder: outs, self.is_train: False}

        step_loss, step_acc = sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
        summary_str = sess.run(self.summary, feed_dict=feed_dict)
        return step_loss, step_acc, summary_str
        # val_list = config.val_list
        # start_index = randint(0,len(val_list)-(config.batches_per_epoch_val+1))
        # pre_scores = []
        # acc_scores = []
        # rec_scores = []
        # count = 0
        # for file_name in val_list[start_index:start_index+config.batches_per_epoch_val]:
        #     pre, acc, rec = self.validate_file(file_name, sess)
        #     pre_scores.append(pre)
        #     acc_scores.append(acc)
        #     rec_scores.append(rec)
        #     count+=1
        #     utils.progress(count, config.batches_per_epoch_val, suffix='validation done')
        # pre_score = np.array(pre_scores).mean()
        # acc_score = np.array(acc_scores).mean()
        # rec_score = np.array(rec_scores).mean()
        return pre_score, acc_score, rec_score

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
            self.pho_logits = modules.phone_network(input_placeholder)
            self.pho_classes = tf.argmax(pho_logits, axis=-1)
            self.pho_probs = tf.nn.softmax(pho_logits)

        with tf.variable_scope('Final_Model') as scope:
            self.voc_output = modules.final_net(singer_onehot_labels, f0_input_placeholder, phone_onehot_labels)
            self.voc_output_decoded = tf.nn.sigmoid(voc_output)
            self.scope.reuse_variables()
            self.oc_output_3 = modules.final_net(singer_onehot_labels, f0_input_placeholder, pho_probs)
            self.voc_output_3_decoded = tf.nn.sigmoid(voc_output_3)

        # with tf.variable_scope('singer_Model') as scope:
        #     singer_embedding, singer_logits = modules.singer_network(input_placeholder, prob)
        #     singer_classes = tf.argmax(singer_logits, axis=-1)
        #     singer_probs = tf.nn.softmax(singer_logits)

        with tf.variable_scope('Generator') as scope:
            self.voc_output_2 = modules.GAN_generator(singer_onehot_labels, phone_onehot_labels, f0_input_placeholder,
                                                 rand_input_placeholder)

        with tf.variable_scope('Discriminator') as scope:
            self.D_real = modules.GAN_discriminator((output_placeholder - 0.5) * 2, singer_onehot_labels,
                                               phone_onehot_labels, f0_input_placeholder)
            scope.reuse_variables()
            self.D_fake = modules.GAN_discriminator(voc_output_2, singer_onehot_labels, phone_onehot_labels,
                                               f0_input_placeholder)



class Voc_Sep(Model):

    def get_placeholders(self):
        """
        Returns the placeholders for the model. 
        Depending on the mode, can return placeholders for either just the generator or both the generator and discriminator.
        """

        self.input_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size, config.max_phr_len, 360),name='input_placeholder')
        self.f0_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size, config.max_phr_len, 360),name='f0_placeholder')
        self.feats_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size, config.max_phr_len, 64),name='feats_placeholder')
        self.is_train = tf.placeholder(tf.bool, name="is_train")

    def loss_function(self):
        """
        returns the loss function for the model, based on the mode. 
        """
        self.loss = tf.reduce_sum(tf.abs(self.output_logits - self.feats_placeholder))

    def read_input_file(self, file_name):
        """
        Function to read and process input file, given name and the synth_mode.
        Returns features for the file based on mode (0 for hdf5 file, 1 for wav file).
        Currently, only the HDF5 version is implemented.
        """
        feat_file = h5py.File(config.feats_dir + file_name, 'r')

        cqt = abs(feat_file['voc_cqt'][()])

        feat_file.close()

        return cqt


    def extract_part_from_file(self, file_name, part, sess):

        parts = ['_soprano_', '_alto_', '_bass_', '_tenor_']

        cqt = self.read_input_file(file_name)

        song_name = file_name.split('_')[0]

        voc_num = 9-part
        voc_part = parts[part]
        voc_track= file_name[-voc_num]

        voc_feat_file = h5py.File(config.voc_feats_dir + song_name + voc_part + voc_track + '.wav.hdf5', 'r')

        voc_feats = voc_feat_file["voc_feats"][()]

        voc_feats[np.argwhere(np.isnan(voc_feats))] = 0.0

        atb = voc_feat_file['atb'][()]

        atb = atb[:, 1:]

        atb[:, 0:4] = 0

        atb = np.clip(atb, 0.0, 1.0)

        max_len = min(len(voc_feats), len(cqt))

        voc_feats = voc_feats[:max_len]

        cqt = cqt[:max_len]

        atb = atb[:max_len]

        # voc_feats = (voc_feats - min_feat) / (max_feat - min_feat)
        #
        # voc_feats = np.clip(voc_feats[:, :, :-2], 0.0, 0.1)

        # sig_process.feats_to_audio(voc_feats, 'booboo.wav')

        in_batches_cqt, nchunks_in = utils.generate_overlapadd(cqt)

        in_batches_atb, nchunks_in = utils.generate_overlapadd(atb)

        # import pdb;pdb.set_trace()
        out_batches_feats = []
        for in_batch_cqt, in_batch_atb in zip(in_batches_cqt, in_batches_atb):
            feed_dict = {self.input_placeholder: in_batch_cqt, self.f0_placeholder: in_batch_atb, self.is_train: False}
            out_feat = sess.run(self.output_logits, feed_dict=feed_dict)
            out_batches_feats.append(out_feat)


        out_batches_feats = np.array(out_batches_feats)


        out_feats = utils.overlapadd(out_batches_feats.reshape(out_batches_feats.shape[0], config.batch_size, config.max_phr_len, -1),
                         nchunks_in)

        out_feats = out_feats * (max_feat - min_feat) + min_feat

        out_feats = out_feats[:max_len]

        out_feats = np.concatenate((out_feats, voc_feats[:, -2:]), axis=-1)

        plt.figure(1)
        plt.subplot(211)
        plt.imshow(voc_feats.T, origin = 'lower', aspect = 'auto')
        plt.subplot(212)
        plt.imshow(out_feats.T, origin='lower', aspect='auto')
        plt.show()


        sig_process.feats_to_audio(out_feats, 'extracted.wav')


        import pdb;pdb.set_trace()

        # import pdb;pdb.set_trace()

    def extract_file(self, file_name, part):
        sess = tf.Session()
        self.load_model(sess, config.log_dir_sep)
        self.extract_part_from_file(file_name, part, sess)

    def train(self):
        """
        Function to train the model, and save Tensorboard summary, for N epochs. 
        """
        sess = tf.Session()


        self.loss_function()
        self.get_optimizers()
        self.load_model(sess, config.log_dir_sep)
        self.get_summary(sess, config.log_dir_sep)
        start_epoch = int(sess.run(tf.train.get_global_step()) / (config.batches_per_epoch_train))


        print("Start from: %d" % start_epoch)


        for epoch in range(start_epoch, config.num_epochs):
            data_generator = sep_gen()
            start_time = time.time()


            batch_num = 0
            epoch_train_loss = 0


            with tf.variable_scope('Training'):
                for ins, f0s, feats in data_generator:

                    step_loss, summary_str = self.train_model(ins, f0s, feats, sess)
                    if np.isnan(step_loss):
                        import pdb;pdb.set_trace()
                    epoch_train_loss+=step_loss

                    self.train_summary_writer.add_summary(summary_str, epoch)
                    self.train_summary_writer.flush()

                    utils.progress(batch_num,config.batches_per_epoch_train, suffix = 'training done')

                    batch_num+=1
                    # import pdb;pdb.set_trace()

                epoch_train_loss = epoch_train_loss/batch_num
                print_dict = {"Training Loss": epoch_train_loss}

            # if (epoch + 1) % config.validate_every == 0:
                # pre, acc, rec = self.validate_model(sess)
                # print_dict["Validation Precision"] = pre
                # print_dict["Validation Accuracy"] = acc
                # print_dict["Validation Recall"] = rec

            end_time = time.time()
            if (epoch + 1) % config.print_every == 0:
                self.print_summary(print_dict, epoch, end_time-start_time)
            if (epoch + 1) % config.save_every == 0 or (epoch + 1) == config.num_epochs:
                self.save_model(sess, epoch+1, config.log_dir_sep)

    def train_model(self, ins, f0s, feats, sess):
        """
        Function to train the model for each epoch
        """
        feed_dict = {self.input_placeholder: ins, self.f0_placeholder: f0s, self.feats_placeholder: feats, self.is_train: False}
        _, step_loss= sess.run(
            [self.train_function, self.loss], feed_dict=feed_dict)
        summary_str = sess.run(self.summary, feed_dict=feed_dict)

        return step_loss, summary_str

    def validate_model(self, sess):
        """
        Function to train the model for each epoch
        """
        # feed_dict = {self.input_placeholder: ins, self.output_placeholder: outs, self.is_train: False}
        #
        # step_loss= sess.run(self.loss, feed_dict=feed_dict)
        # summary_str = sess.run(self.summary, feed_dict=feed_dict)
        # return step_loss, summary_str
        val_list = config.val_list
        start_index = randint(0,len(val_list)-(config.batches_per_epoch_val+1))
        pre_scores = []
        acc_scores = []
        rec_scores = []
        count = 0
        for file_name in val_list[start_index:start_index+config.batches_per_epoch_val]:
            pre, acc, rec = self.validate_file(file_name, sess)
            pre_scores.append(pre)
            acc_scores.append(acc)
            rec_scores.append(rec)
            count+=1
            utils.progress(count, config.batches_per_epoch_val, suffix='validation done')
        pre_score = np.array(pre_scores).mean()
        acc_score = np.array(acc_scores).mean()
        rec_score = np.array(rec_scores).mean()
        return pre_score, acc_score, rec_score

    def eval_all(self, file_name_csv):
        sess = tf.Session()
        self.load_model(sess)
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
        scores = pd.DataFrame.from_dict(scores)
        scores.to_csv(file_name_csv)

        return scores


    def model(self):
        """
        The main model function, takes and returns tensors.
        Defined in modules.

        """
        with tf.variable_scope('Model') as scope:
            self.output_logits = nr_wavenet(self.input_placeholder,self.f0_placeholder, self.is_train)

def test():
    # model = DeepSal()
    # # model.test_file('nino_4424.hdf5')
    # model.test_wav_folder('./helena_test_set/', './results/')

    model = Voc_Sep()
    model.extract_file('locus_0024.hdf5', 3)

if __name__ == '__main__':
    test()





