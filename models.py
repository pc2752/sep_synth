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

        self.final_optimizer = tf.train.AdamOptimizer(learning_rate = config.init_lr)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)



        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.final_train_function = self.final_optimizer.minimize(self.final_loss, global_step = self.global_step, var_list = self.final_params)


    def loss_function(self):
        """
        returns the loss function for the model, based on the mode. 
        """

        # self.final_loss = tf.losses.mean_squared_error(self.output,self.output_placeholder )
        self.final_loss = tf.reduce_sum(tf.abs(self.output_placeholder- self.output))


    def get_summary(self, sess, log_dir):
        """
        Gets the summaries and summary writers for the losses.
        """


        self.final_summary = tf.summary.scalar('final_loss', self.final_loss)

        self.train_summary_writer = tf.summary.FileWriter(log_dir+'train/', sess.graph)
        self.val_summary_writer = tf.summary.FileWriter(log_dir+'val/', sess.graph)
        self.summary = tf.summary.merge_all()

    def get_placeholders(self):
        """
        Returns the placeholders for the model. 
        Depending on the mode, can return placeholders for either just the generator or both the generator and discriminator.
        """

        self.input_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size, config.max_phr_len, config.output_features),
                                           name='input_placeholder')

        self.cond_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size, config.max_phr_len, config.input_features),
                                           name='cond_placeholder')

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

            val_final_loss = 0

            with tf.variable_scope('Training'):
                for conds, voc_out in data_generator:

                    voc_in = np.roll(voc_out, 1, 1)

                    voc_in[:,0,:] = 0

                    voc_in = voc_in + np.random.normal(0,.5,(voc_in.shape)) * 0.4 

                    final_loss, summary_str = self.train_model(conds, voc_out, voc_in, sess)


                    epoch_final_loss+=final_loss


                    self.train_summary_writer.add_summary(summary_str, epoch)
                    self.train_summary_writer.flush()

                    utils.progress(batch_num,config.batches_per_epoch_train, suffix = 'training done')

                    batch_num+=1

                epoch_final_loss = epoch_final_loss/batch_num


                print_dict = {"Final Loss": epoch_final_loss}


            if (epoch + 1) % config.validate_every == 0:
                batch_num = 0
                with tf.variable_scope('Validation'):
                    for conds, voc_out in val_generator:

                        voc_in = np.roll(voc_out, 1, 1)

                        voc_in[:,0,:] = 0

                        final_loss, summary_str= self.validate_model(conds, voc_out, voc_in, sess)
                        val_final_loss+=final_loss


                        self.val_summary_writer.add_summary(summary_str, epoch)
                        self.val_summary_writer.flush()
                        batch_num+=1

                        utils.progress(batch_num, config.batches_per_epoch_val, suffix='validation done')

                    val_final_loss = val_final_loss/batch_num


                    print_dict["Val Final Loss"] =  val_final_loss


            end_time = time.time()
            if (epoch + 1) % config.print_every == 0:
                self.print_summary(print_dict, epoch, end_time-start_time)
            if (epoch + 1) % config.save_every == 0 or (epoch + 1) == config.num_epochs:
                self.save_model(sess, epoch+1, config.log_dir)

    def train_model(self, conds, voc_out, voc_in,sess):
        """
        Function to train the model for each epoch
        """
        feed_dict = {self.input_placeholder: voc_in, self.output_placeholder: voc_out, self.cond_placeholder: conds, self.is_train: True}

        _,final_loss= sess.run([self.final_train_function, self.final_loss], feed_dict=feed_dict)

        summary_str = sess.run(self.summary, feed_dict=feed_dict)

        return final_loss, summary_str

    def validate_model(self,conds, voc_out,voc_in, sess):
        """
        Function to train the model for each epoch
        """
        feed_dict = {self.input_placeholder: voc_in, self.output_placeholder: voc_out, self.cond_placeholder: conds, self.is_train: False}

        final_loss= sess.run(self.final_loss, feed_dict=feed_dict)

        summary_str = sess.run(self.summary, feed_dict=feed_dict)

        return final_loss, summary_str



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

        
        return np.clip(mix_stft, 0.0, 1.0), feats

    def test_file_hdf5(self, file_name):
        """
        Function to extract multi pitch from file. Currently supports only HDF5 files.
        """
        sess = tf.Session()
        self.load_model(sess, log_dir = config.log_dir)
        condi, feats = self.read_hdf5_file(file_name)

        out_feats = self.process_file(condi, sess)

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


    def process_file(self, condi, sess):

        stat_file = h5py.File(config.stat_dir+'stats.hdf5', mode='r')

        max_feat = np.array(stat_file["feats_maximus"])
        min_feat = np.array(stat_file["feats_minimus"])
        stat_file.close()

        conds = np.zeros((config.batch_size, config.max_phr_len, config.input_features))
        outs = []
        

        in_batches_stft, nchunks_in = utils.generate_overlapadd(condi)

        out_batches = []

        for in_batch_stft in in_batches_stft :
            outs = []
            inps = np.zeros((config.batch_size, config.max_phr_len, config.output_features))
            for i in range(config.max_phr_len):
                feed_dict = {self.cond_placeholder: in_batch_stft, self.input_placeholder: inps, self.is_train: False}
                harm = sess.run(self.output, feed_dict=feed_dict)
                outs.append(harm[:,-1,:])
                inps = np.roll(inps, -1, 1)
                inps[:,-1,:] = harm[:,-1,:]
            outs = np.swapaxes(np.array(outs), 0,1)
            out_batches.append(outs)
        out_batches = np.array(out_batches)
        out_batches = utils.overlapadd(out_batches,nchunks_in)
        outs = outs*(max_feat - min_feat) + min_feat
        return np.array(outs)



    def model(self):
        """
        The main model function, takes and returns tensors.
        Defined in modules.

        """

        with tf.variable_scope('Final_Model') as scope:
            self.output = modules.wave_archi(self.input_placeholder, self.cond_placeholder, self.is_train)



def test():
    # model = DeepSal()
    # # model.test_file('nino_4424.hdf5')
    # model.test_wav_folder('./helena_test_set/', './results/')

    model = MultiSynth()
    model.train()

if __name__ == '__main__':
    test()





