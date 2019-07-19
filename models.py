import tensorflow as tf
import modules_tf as modules
import config
from data_pipeline import data_gen_sep
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


def one_hotize(inp, max_index):


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
            

class SepNet(Model):

    def get_optimizers(self):
        """
        Returns the optimizers for the model, based on the loss functions and the mode. 
        """

        self.final_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'Final_Model')
        self.f0_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'F0_Model')

        self.final_optimizer = tf.train.AdamOptimizer(learning_rate = config.init_lr)
        self.f0_optimizer = tf.train.AdamOptimizer(learning_rate = config.init_lr)
        
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.global_step_f0 = tf.Variable(0, name='f0_global_step', trainable=False)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.final_train_function = self.final_optimizer.minimize(self.final_loss, global_step = self.global_step, var_list = self.final_params)
            self.f0_train_function = self.f0_optimizer.minimize(self.f0_loss, global_step = self.global_step_f0, var_list = self.f0_params)


    def loss_function(self):
        """
        returns the loss function for the model, based on the mode. 
        """

        # self.final_loss = tf.losses.mean_squared_error(self.output,self.output_placeholder )
        self.final_loss = tf.reduce_sum(tf.abs(self.output_placeholder- self.output))

        self.f0_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.f0_placeholder_onehot, logits=self.f0_logits))

        self.f0_acc = tf.metrics.accuracy(labels=self.f0_placeholder, predictions=self.f0_classes)

    def get_summary(self, sess, log_dir):
        """
        Gets the summaries and summary writers for the losses.
        """


        self.final_summary = tf.summary.scalar('final_loss', self.final_loss)
        self.f0_summary = tf.summary.scalar('f0_loss', self.f0_loss)

        self.f0_acc_summary = tf.summary.scalar('f0_accuracy', self.f0_acc[0])

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

        self.f0_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size, config.max_phr_len),
                                           name='f0_placeholder')

        self.f0_placeholder_onehot = tf.placeholder(tf.float32, shape=(config.batch_size, config.max_phr_len, config.num_f0),
                                           name='f0_placeholder')

        # self.f0_placeholder_onehot = tf.one_hot(indices=tf.cast(self.f0_placeholder, tf.int32), depth=config.num_f0)

        self.output_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size, config.max_phr_len, config.output_features),
                                           name='output_placeholder')       

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
            data_generator = data_gen_sep()
            val_generator = data_gen_sep(mode = 'Val')
            start_time = time.time()


            batch_num = 0
            epoch_final_loss = 0

            epoch_f0_loss = 0
            epoch_f0_acc = 0

            val_final_loss = 0
            val_f0_loss = 0
            val_f0_acc = 0

            with tf.variable_scope('Training'):
                for conds, voc_out, f0_out in data_generator:


                    final_loss, f0_loss, f0_acc, summary_str = self.train_model(conds, voc_out, f0_out, epoch, sess)


                    epoch_final_loss+=final_loss
                    epoch_f0_loss+=f0_loss
                    epoch_f0_acc+=f0_acc

                    self.train_summary_writer.add_summary(summary_str, epoch)
                    self.train_summary_writer.flush()

                    utils.progress(batch_num,config.batches_per_epoch_train, suffix = 'training done')

                    batch_num+=1

                epoch_final_loss = epoch_final_loss/batch_num

                epoch_f0_loss = epoch_f0_loss/batch_num
                epoch_f0_acc = epoch_f0_acc/batch_num


                print_dict = {"Final Loss": epoch_final_loss}
                print_dict["F0 Loss"] =  epoch_f0_loss
                print_dict["F0 Accuracy"] =  epoch_f0_acc

            if (epoch + 1) % config.validate_every == 0:
                batch_num = 0
                with tf.variable_scope('Validation'):
                    for conds, voc_out, f0_out in val_generator:

                        final_loss, f0_loss, f0_acc, summary_str = self.validate_model(conds, voc_out, f0_out, sess)
                        val_final_loss+=final_loss

                        val_f0_loss+=f0_loss
                        val_f0_acc+=f0_acc

                        self.val_summary_writer.add_summary(summary_str, epoch)
                        self.val_summary_writer.flush()
                        batch_num+=1

                        utils.progress(batch_num, config.batches_per_epoch_val, suffix='validation done')

                    val_final_loss = val_final_loss/batch_num
                    val_f0_loss = val_f0_loss/batch_num
                    val_f0_acc = val_f0_acc/batch_num

                    print_dict["Val Final Loss"] =  val_final_loss
                    print_dict["Val F0 Loss"] =  val_f0_loss
                    print_dict["Val F0 Accuracy"] =  val_f0_acc


            end_time = time.time()
            if (epoch + 1) % config.print_every == 0:
                self.print_summary(print_dict, epoch, end_time-start_time)
            if (epoch + 1) % config.save_every == 0 or (epoch + 1) == config.num_epochs:
                self.save_model(sess, epoch+1, config.log_dir)

    def train_model(self, conds, voc_out, f0_out, epoch, sess):
        """
        Function to train the model for each epoch
        """
        teach_train = np.random.rand(1)<0.5

        if epoch<700 or not teach_train:

            feed_dict = {self.input_placeholder: conds, self.f0_placeholder_onehot: f0_out, self.output_placeholder: voc_out, self.f0_placeholder: np.argmax(f0_out, axis = -1), self.is_train: True}

            _,_,final_loss, f0_loss, f0_acc = sess.run([self.final_train_function, self.f0_train_function, self.final_loss, self.f0_loss, self.f0_acc], feed_dict=feed_dict)
        else:
            feed_dict = {self.input_placeholder: conds, self.f0_placeholder_onehot: f0_out, self.output_placeholder: voc_out, self.f0_placeholder: np.argmax(f0_out, axis = -1), self.is_train: True}
            _, f0_loss, f0_acc = sess.run([self.f0_train_function, self.f0_loss, self.f0_acc], feed_dict=feed_dict)

            f0_in = sess.run(self.f0_probs, feed_dict = {self.input_placeholder: conds, self.is_train: False})

            feed_dict = {self.input_placeholder: conds, self.f0_placeholder_onehot: f0_in, self.output_placeholder: voc_out, self.is_train: True}

            _,final_loss = sess.run([self.final_train_function, self.final_loss], feed_dict=feed_dict)

        summary_str = sess.run(self.summary, feed_dict=feed_dict)

        return final_loss, f0_loss, f0_acc[0], summary_str

    def validate_model(self,conds, voc_out,f0_out, sess):
        """
        Function to train the model for each epoch
        """
        feed_dict = {self.input_placeholder: conds, self.f0_placeholder_onehot: f0_out, self.output_placeholder: voc_out, self.f0_placeholder: np.argmax(f0_out, axis = -1), self.is_train: False}

        final_loss, f0_loss, f0_acc = sess.run([self.final_loss, self.f0_loss, self.f0_acc], feed_dict=feed_dict)

        summary_str = sess.run(self.summary, feed_dict=feed_dict)

        return final_loss, f0_loss, f0_acc[0], summary_str



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
        stat_file.close()

        with h5py.File(config.voice_dir + file_name) as feat_file:
            feats = np.array(feat_file['feats'])

        with h5py.File(config.backing_dir + file_name) as feat_file:
            mix_stft = np.array(feat_file["mix_stft"])

        f0 = feats[:,-2]

        med = np.median(f0[f0 > 0])

        f0[f0==0] = med

        f0_nor = (f0 - min_feat[-2])/(max_feat[-2]-min_feat[-2])

        f0_quant = np.rint(f0_nor*(config.num_f0-2)) + 1

        f0_quant = f0_quant * (1-feats[:,-1]) 

        f0_quant = one_hotize(f0_quant, config.num_f0)

        return np.clip(mix_stft, 0.0, 1.0), feats, f0_quant

    def test_file_hdf5(self, file_name):
        """
        Function to extract multi pitch from file. Currently supports only HDF5 files.
        """
        sess = tf.Session()
        self.load_model(sess, log_dir = config.log_dir)
        condi, feats, f0_quant = self.read_hdf5_file(file_name)

        out_feats = self.process_file(condi, f0_quant, sess)

        self.plot_features(feats, out_feats)

        import pdb;pdb.set_trace()

        out_featss = np.concatenate((out_feats[:feats.shape[0]], feats[:,-2:]), axis = -1)

        utils.feats_to_audio(out_featss,file_name[:-4]+'_'+str(config.singers[singer_id])+'.wav') 

    def plot_features(self, feats, out_feats):

        plt.figure(1)
        
        ax1 = plt.subplot(211)

        plt.imshow(feats[:,:-2].T,aspect='auto',origin='lower')

        ax1.set_title("Ground Truth Vocoder Features", fontsize=10)

        ax3 =plt.subplot(212, sharex = ax1, sharey = ax1)

        ax3.set_title("Output STFT", fontsize=10)

        plt.imshow(out_feats[:,:-2].T,aspect='auto',origin='lower')
        
        plt.figure(2)

        f0_output = out_feats[:feats.shape[0],-2]
        f0_output = f0_output*(1-feats[:,-1])
        f0_output[f0_output == 0] = np.nan
        plt.plot(f0_output, label = "Predicted Value")
        f0_gt = feats[:,-2]
        f0_gt = f0_gt*(1-feats[:,-1])
        f0_gt[f0_gt == 0] = np.nan
        plt.plot(f0_gt, label="Ground Truth")
        f0_difference = np.nan_to_num(abs(f0_gt-f0_output))
        f0_greater = np.where(f0_difference>config.f0_threshold)
        diff_per = f0_greater[0].shape[0]/len(f0_output)
        plt.suptitle("Percentage correct = "+'{:.3%}'.format(1-diff_per))

        plt.show()


    def process_file(self, condi, f0_quant, sess):

        stat_file = h5py.File(config.stat_dir+'stats.hdf5', mode='r')

        max_feat = np.array(stat_file["feats_maximus"])
        min_feat = np.array(stat_file["feats_minimus"])
        stat_file.close()

        conds = np.zeros((config.batch_size, config.max_phr_len, config.input_features))
        outs = []
        

        in_batches_stft, nchunks_in = utils.generate_overlapadd(condi)
        in_batches_f0, nchunks_in = utils.generate_overlapadd(f0_quant)

        out_batches = []

        out_f0 = []

        for in_batch_stft, in_f0 in zip(in_batches_stft, in_batches_f0) :
            f0_in = sess.run(self.f0_probs, feed_dict = {self.input_placeholder: in_batch_stft, self.is_train: False})
            out_f0.append(f0_in)
            feed_dict = {self.input_placeholder: in_batch_stft, self.f0_placeholder_onehot: in_f0, self.is_train: False}

            output = sess.run(self.output, feed_dict = feed_dict)
            out_batches.append(output)

        out_batches = np.array(out_batches) *(max_feat-min_feat)+min_feat
        out_batches = utils.overlapadd(out_batches, nchunks_in) 
        out_f0 = utils.overlapadd(np.array(out_f0), nchunks_in) 

        return out_batches



    def model(self):
        """
        The main model function, takes and returns tensors.
        Defined in modules.

        """
        with tf.variable_scope('F0_Model') as scope:
            self.f0_logits = modules.f0_network(self.input_placeholder, self.is_train)
            self.f0_classes = tf.argmax(self.f0_logits, axis=-1)
            self.f0_probs = tf.nn.softmax(self.f0_logits)

        with tf.variable_scope('Final_Model') as scope:
            self.output = modules.full_network(self.input_placeholder, self.f0_placeholder_onehot, self.is_train)



def test():
    # model = DeepSal()
    # # model.test_file('nino_4424.hdf5')
    # model.test_wav_folder('./helena_test_set/', './results/')

    model = MultiSynth()
    model.train()

if __name__ == '__main__':
    test()





