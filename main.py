import tensorflow as tf


import matplotlib.pyplot as plt

import os
import sys

# sys.path.insert(0, './griffin_lim/')
# import audio_utilities
import time
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import h5py
import soundfile as sf
import config
from data_pipeline import data_gen
import modules_tf as modules
import utils
from reduce import mgc_to_mfsc
import models

def one_hotize(inp, max_index=41):
    # output = np.zeros((inp.shape[0],inp.shape[1],max_index))
    # for i, index in enumerate(inp):
    #     output[i,index] = 1
    # import pdb;pdb.set_trace()
    output = np.eye(max_index)[inp.astype(int)]
    # import pdb;pdb.set_trace()
    # output = np.eye(max_index)[inp]
    return output
def binary_cross(p,q):
    return -(p * tf.log(q + 1e-12) + (1 - p) * tf.log( 1 - q + 1e-12))



def train(_):
    stat_file = h5py.File(config.stat_dir+'stats.hdf5', mode='r')
    max_feat = np.array(stat_file["feats_maximus"])
    min_feat = np.array(stat_file["feats_minimus"])
    with tf.Graph().as_default():
        
        input_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size,config.max_phr_len,513),name='input_placeholder')
        tf.summary.histogram('inputs', input_placeholder)

        output_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size,config.max_phr_len,513),name='output_placeholder')


        f0_input_placeholder= tf.placeholder(tf.float32, shape=(config.batch_size,config.max_phr_len, 1),name='f0_input_placeholder')

        # rand_input_placeholder= tf.placeholder(tf.float32, shape=(config.batch_size,1,1, 4),name='rand_input_placeholder')


        # pho_input_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size,config.max_phr_len, 42),name='pho_input_placeholder')

        prob = tf.placeholder_with_default(1.0, shape=())
        
        phoneme_labels = tf.placeholder(tf.int32, shape=(config.batch_size,config.max_phr_len),name='phoneme_placeholder')
        phone_onehot_labels = tf.one_hot(indices=tf.cast(phoneme_labels, tf.int32), depth=42)

        singer_labels = tf.placeholder(tf.float32, shape=(config.batch_size),name='singer_placeholder')
        singer_onehot_labels = tf.one_hot(indices=tf.cast(singer_labels, tf.int32), depth=12)


        with tf.variable_scope('phone_Model') as scope:
            # regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
            pho_logits, phone_embeding = modules.phone_network(input_placeholder)
            pho_classes = tf.argmax(pho_logits, axis=-1)
            pho_probs = tf.nn.softmax(pho_logits)

        # with tf.variable_scope('Final_Model') as scope:
        #     voc_output = modules.final_net(singer_onehot_labels, f0_input_placeholder, phone_onehot_labels)
        #     voc_output_decoded = tf.nn.sigmoid(voc_output)
        #     scope.reuse_variables()
        #     voc_output_3 = modules.final_net(singer_onehot_labels, f0_input_placeholder, pho_probs)
        #     voc_output_3_decoded = tf.nn.sigmoid(voc_output_3)
            

        with tf.variable_scope('singer_Model') as scope:
            singer_embedding, singer_logits = modules.singer_network(input_placeholder)
            singer_classes = tf.argmax(singer_logits, axis=-1)
            singer_probs = tf.nn.softmax(singer_logits)


        with tf.variable_scope('Generator') as scope: 
            voc_output_2 = modules.GAN_generator(singer_embedding, phone_embeding, f0_input_placeholder)
            # scope.reuse_variables()
            # voc_output_2_2 = modules.GAN_generator(voc_output_3_decoded, singer_onehot_labels, phone_onehot_labels, f0_input_placeholder, rand_input_placeholder)


        with tf.variable_scope('phone_Model') as scope:
            scope.reuse_variables()
            pho_logits_2, phone_embeding_2 = modules.phone_network((voc_output_2/2+0.5))
            pho_classes_2 = tf.argmax(pho_logits_2, axis=-1)
            pho_probs_2 = tf.nn.softmax(pho_logits_2)

        with tf.variable_scope('singer_Model') as scope:
            scope.reuse_variables()
            singer_embedding_2, singer_logits_2 = modules.singer_network((voc_output_2/2+0.5))
            singer_classes_2 = tf.argmax(singer_logits, axis=-1)
            singer_probs_2 = tf.nn.softmax(singer_logits)

        with tf.variable_scope('Discriminator') as scope: 
            D_real = modules.GAN_discriminator((output_placeholder-0.5)*2)
            scope.reuse_variables()
            
            D_fake = modules.GAN_discriminator(voc_output_2)


        # import pdb;pdb.set_trace()


        # Get network parameters

        final_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="Final_Model")

        g_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="Generator")

        d_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="Discriminator")

        phone_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="phone_Model")



        # Phoneme network loss and summary

        pho_weights = tf.reduce_sum(config.phonemas_weights * phone_onehot_labels, axis=-1)

        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=phone_onehot_labels, logits=pho_logits)

        weighted_losses = unweighted_losses * pho_weights

        pho_loss = tf.reduce_mean(weighted_losses) 

        # pho_loss_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=phone_onehot_labels, logits=pho_logits_2))
        # +tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels= output_placeholder, logits=voc_output_3))*0.001 

        # reconstruct_loss_pho = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels = output_placeholder, logits=voc_output_decoded_gen)) *0.00001

        # pho_loss+=reconstruct_loss_pho

        pho_acc = tf.metrics.accuracy(labels=phoneme_labels, predictions=pho_classes)

        pho_summary = tf.summary.scalar('pho_loss', pho_loss)

        pho_acc_summary = tf.summary.scalar('pho_accuracy', pho_acc[0])

        singer_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=singer_onehot_labels, logits=singer_logits))

        singer_acc = tf.metrics.accuracy(labels=singer_labels , predictions=singer_classes)

        singer_acc_val = tf.metrics.accuracy(labels=singer_labels , predictions=singer_classes)


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

        # D_correct_pred_fake = tf.equal(tf.round(tf.sigmoid(D_fake_real)), tf.ones_like(D_fake_real))

        D_accuracy = tf.reduce_mean(tf.cast(D_correct_pred, tf.float32))

        # D_accuracy_fake = tf.reduce_mean(tf.cast(D_correct_pred_fake, tf.float32))



        D_loss = tf.reduce_mean(D_real +1e-12) - tf.reduce_mean(D_fake+1e-12)
        # -tf.reduce_mean(D_fake_real+1e-12)*0.001

        dis_summary = tf.summary.scalar('dis_loss', D_loss)

        dis_acc_summary = tf.summary.scalar('dis_acc', D_accuracy)

        # dis_acc_fake_summary = tf.summary.scalar('dis_acc_fake', D_accuracy_fake)

        #Final net loss

        # G_loss_GAN = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels= tf.ones_like(D_real), logits=D_fake+1e-12)) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels= tf.ones_like(D_fake_2), logits=D_fake_2+1e-12))
        # + tf.reduce_sum(tf.abs(output_placeholder- (voc_output_2/2+0.5))*(1-input_placeholder[:,:,-1:])) *0.00001

        G_loss_GAN = tf.reduce_mean(D_fake+1e-12) + tf.reduce_sum(tf.abs(output_placeholder- (voc_output_2/2+0.5)))/(config.batch_size*config.max_phr_len*64)+tf.reduce_mean(tf.abs(phone_embeding- phone_embeding_2)) + tf.reduce_mean(tf.abs(singer_embedding - singer_embedding_2))
                     # + tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels= output_placeholder, logits=voc_output)) *0.000005
        #

        G_correct_pred = tf.equal(tf.round(tf.sigmoid(D_fake)), tf.ones_like(D_real))

        # G_correct_pred_2 = tf.equal(tf.round(tf.sigmoid(D_fake_2)), tf.ones_like(D_real))

        G_accuracy = tf.reduce_mean(tf.cast(G_correct_pred, tf.float32))

        gen_summary = tf.summary.scalar('gen_loss', G_loss_GAN)

        gen_acc_summary = tf.summary.scalar('gen_acc', G_accuracy)

        # final_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels= output_placeholder, logits=voc_output)) \
                           # +tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels= output_placeholder, logits=voc_output_3))*0.5

        # reconstruct_loss = tf.reduce_sum(tf.abs(output_placeholder- (voc_output_2/2+0.5)))


        # final_summary = tf.summary.scalar('final_loss', final_loss)

        summary = tf.summary.merge_all()

        # summary_val = tf.summary.merge([f0_summary_midi, pho_summary, singer_summary, reconstruct_summary, pho_acc_summary_val,  f0_acc_summary_midi_val, singer_acc_summary_val ])

        # vuv_summary = tf.summary.scalar('vuv_loss', vuv_loss)

        # loss_summary = tf.summary.scalar('total_loss', loss)


        #Global steps

        global_step = tf.Variable(0, name='global_step', trainable=False)

        # global_step_re = tf.Variable(0, name='global_step_re', trainable=False)

        global_step_dis = tf.Variable(0, name='global_step_dis', trainable=False)

        global_step_gen = tf.Variable(0, name='global_step_gen', trainable=False)

        global_step_singer = tf.Variable(0, name='global_step_singer', trainable=False)




        #Optimizers

        pho_optimizer = tf.train.AdamOptimizer(learning_rate = config.init_lr)

        # re_optimizer = tf.train.AdamOptimizer(learning_rate = config.init_lr)

        singer_optimizer = tf.train.AdamOptimizer(learning_rate = config.init_lr)

        dis_optimizer = tf.train.RMSPropOptimizer(learning_rate=5e-5)

        gen_optimizer = tf.train.RMSPropOptimizer(learning_rate=5e-5)
        # GradientDescentOptimizer



        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # Training functions
        pho_train_function = pho_optimizer.minimize(pho_loss, global_step = global_step, var_list = phone_params)

        singer_train_function = singer_optimizer.minimize(singer_loss, global_step = global_step_singer)

        # with tf.control_dependencies(update_ops):
        # re_train_function = re_optimizer.minimize(final_loss, global_step = global_step_re, var_list=final_params)

        dis_train_function = dis_optimizer.minimize(D_loss, global_step = global_step_dis, var_list=d_params)

        gen_train_function = gen_optimizer.minimize(G_loss_GAN, global_step = global_step_gen, var_list=g_params)

        clip_discriminator_var_op = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in d_params]

        

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        saver = tf.train.Saver(max_to_keep= config.max_models_to_keep)
        sess = tf.Session()

        sess.run(init_op)

        ckpt = tf.train.get_checkpoint_state(config.log_dir)

        if ckpt and ckpt.model_checkpoint_path:
            print("Using the model in %s"%ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)


        train_summary_writer = tf.summary.FileWriter(config.log_dir+'train/', sess.graph)
        val_summary_writer = tf.summary.FileWriter(config.log_dir+'val/', sess.graph)

        
        start_epoch = int(sess.run(tf.train.get_global_step())/(config.batches_per_epoch_train))

        print("Start from: %d" % start_epoch)
        
        for epoch in xrange(start_epoch, config.num_epochs):

            if epoch<2050 or epoch < 50 or epoch%100 == 0:
                n_critic = 25
            else:
                n_critic = 5
            if epoch<2000:

                data_generator = data_gen(sec_mode = 0)
            else:
                data_generator = data_gen(sec_mode = 1)
            start_time = time.time()

            val_generator = data_gen(mode='val')

            batch_num = 0

            epoch_pho_loss = 0
            epoch_gen_loss = 0
            epoch_re_loss = 0
            epoch_dis_loss = 0

            epoch_pho_acc = 0
            epoch_gen_acc = 0
            epoch_singer_acc = 0


            val_epoch_pho_loss = 0
            val_epoch_gen_loss = 0
            val_epoch_dis_loss = 0

            val_epoch_pho_acc = 0
            val_epoch_gen_acc = 0
            val_epoch_singer_acc = 0

            flag_count = 0


            with tf.variable_scope('Training'):

                for feats, f0, phos, singer_ids, mix_in, Flag in data_generator:



                    # pho_one_hot = one_hotize(phos, max_index=42)

                    f0 = f0.reshape([config.batch_size, config.max_phr_len, 1])

                    if Flag:

                        flag_count+=1

                        assert singer_ids.max()<=11 and singer_ids.min()>=0


                        feed_dict = {input_placeholder: mix_in, output_placeholder: mix_in, f0_input_placeholder: f0,phoneme_labels:phos, singer_labels: singer_ids}

                        _,_, step_pho_loss, step_pho_acc, step_sing_loss, step_sing_acc = sess.run([pho_train_function, singer_train_function, pho_loss, pho_acc, singer_loss, singer_acc], feed_dict= feed_dict)
                        epoch_pho_acc+=step_pho_acc[0]
                        # epoch_gen_acc+=step_gen_acc
                        epoch_singer_acc+=step_sing_acc[0]
                        epoch_pho_loss+=step_pho_loss

                    if epoch>=2000:

                        for critic_itr in range(n_critic):
                            feed_dict = {input_placeholder: mix_in, output_placeholder: feats[:,:,:-2], f0_input_placeholder: f0}
                            sess.run(dis_train_function, feed_dict = feed_dict)
                            sess.run(clip_discriminator_var_op, feed_dict = feed_dict)



                        _, step_gen_loss, step_gen_acc = sess.run([gen_train_function,G_loss_GAN, G_accuracy], feed_dict = feed_dict)
                        # if step_gen_acc>0.3:
                        step_dis_loss, step_dis_acc= sess.run([D_loss, D_accuracy], feed_dict = feed_dict)
                            # else: 
                                # step_dis_loss, step_dis_acc = sess.run([D_loss, D_accuracy], feed_dict = feed_dict)

                            # import pdb;pdb.set_trace()

                            
                            # epoch_re_loss+=step_re_loss
                        epoch_gen_loss+=step_gen_loss
                        epoch_dis_loss+=step_dis_loss





                    utils.progress(batch_num,config.batches_per_epoch_train, suffix = 'training done')
                    batch_num+=1

                if Flag:

                    epoch_pho_acc = epoch_pho_acc/flag_count
                    # epoch_gen_acc = epoch_gen_acc/config.batches_per_epoch_train
                    epoch_singer_acc = epoch_singer_acc/flag_count
                    epoch_pho_loss = epoch_pho_loss/flag_count

                if epoch>=2000:
                    epoch_gen_loss = epoch_gen_loss/config.batches_per_epoch_train
                    epoch_dis_loss = epoch_dis_loss/config.batches_per_epoch_train


# epoch_re_loss = epoch_re_loss/config.batches_per_epoch_train





            #     summary_str = sess.run(summary, feed_dict=feed_dict)
            # # import pdb;pdb.set_trace()
            #     train_summary_writer.add_summary(summary_str, epoch)
            # # # summary_writer.add_summary(summary_str_val, epoch)
            #     train_summary_writer.flush()


            # with tf.variable_scope('Validation'):

            #     for feats, f0, phos, singer_ids, mix_in in val_generator:

            #         pho_one_hot = one_hotize(phos, max_index=42)

            #         f0 = f0.reshape([config.batch_size, config.max_phr_len, 1])

            #         sing_id_shu = np.copy(singer_ids)

            #         phos_shu = np.copy(phos)

            #         np.random.shuffle(sing_id_shu)

            #         np.random.shuffle(phos_shu)

            #         feed_dict = {input_placeholder: mix_in, output_placeholder: feats[:,:,:-2], f0_input_placeholder: f0,rand_input_placeholder: np.random.uniform(-1.0,1.0,size=[30,config.max_phr_len,4]),
            #         phoneme_labels:phos, singer_labels: singer_ids}

            #         step_pho_loss, step_pho_acc = sess.run([pho_loss, pho_acc], feed_dict= feed_dict)
            #         step_gen_loss, step_gen_acc = sess.run([G_loss_GAN, G_accuracy], feed_dict = feed_dict)
            #         step_dis_loss, step_sing_acc = sess.run([D_loss, singer_acc_val], feed_dict = feed_dict)

            #         val_epoch_pho_loss+=step_pho_loss
            #         val_epoch_gen_loss+=step_gen_loss
            #         val_epoch_dis_loss+=step_dis_loss

            #         val_epoch_pho_acc+=step_pho_acc[0]
            #         val_epoch_gen_acc+=step_gen_acc
            #         val_epoch_singer_acc+=step_sing_acc[0]




                    # utils.progress(batch_num,config.batches_per_epoch_train, suffix = 'training done')
                    # batch_num+=1

                # val_epoch_pho_loss = val_epoch_pho_loss/config.batches_per_epoch_val
                # val_epoch_gen_loss = val_epoch_gen_loss/config.batches_per_epoch_val
                # val_epoch_dis_loss = val_epoch_dis_loss/config.batches_per_epoch_val

                # val_epoch_pho_acc = val_epoch_pho_acc/config.batches_per_epoch_val
                # val_epoch_gen_acc = val_epoch_gen_acc/config.batches_per_epoch_val
                # val_epoch_singer_acc = val_epoch_singer_acc/config.batches_per_epoch_val


            #     summary_str = sess.run(summary, feed_dict=feed_dict)
            # # import pdb;pdb.set_trace()
            #     val_summary_writer.add_summary(summary_str, epoch)
            # # # summary_writer.add_summary(summary_str_val, epoch)
            #     val_summary_writer.flush()
            duration = time.time() - start_time

            # np.save('./ikala_eval/accuracies', f0_accs)

            if (epoch+1) % config.print_every == 0:
                print('epoch %d: Phone Loss = %.10f (%.3f sec)' % (epoch+1, epoch_pho_loss, duration))
                print('        : Phone Accuracy = %.10f ' % (epoch_pho_acc))
                # print('        : Recon Loss = %.10f ' % (epoch_re_loss))
                print('        : Gen Loss = %.10f ' % (epoch_gen_loss))
                # print('        : Gen Accuracy = %.10f ' % (epoch_gen_acc))
                print('        : Dis Loss = %.10f ' % (epoch_dis_loss))
                print('        : Singer Accuracy = %.10f ' % (epoch_singer_acc))
                # print('        : Dis Accuracy Fake = %.10f ' % (epoch_dis_acc_fake))
                # print('        : Val Phone Accuracy = %.10f ' % (val_epoch_pho_acc))
                # print('        : Val Gen Loss = %.10f ' % (val_epoch_gen_loss))
                # print('        : Val Gen Accuracy = %.10f ' % (val_epoch_gen_acc))
                # print('        : Val Dis Loss = %.10f ' % (val_epoch_dis_loss))
                # print('        : Val Singer Accuracy = %.10f ' % (val_epoch_singer_acc))

                # import pdb;pdb.set_trace()

            if (epoch + 1) % config.save_every == 0 or (epoch + 1) == config.num_epochs:
                # utils.list_to_file(val_f0_accs,'./ikala_eval/accuracies_'+str(epoch+1)+'.txt')
                checkpoint_file = os.path.join(config.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=epoch)


def synth_file(file_name = "nus_MCUR_sing_10.hdf5", singer_index = 0, file_path=config.wav_dir, show_plots=True, save_file = "GBO"):


    stat_file = h5py.File(config.stat_dir+'stats.hdf5', mode='r')
    max_feat = np.array(stat_file["feats_maximus"])
    min_feat = np.array(stat_file["feats_minimus"])
    with tf.Graph().as_default():
        
        input_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size,config.max_phr_len,513),name='input_placeholder')
        tf.summary.histogram('inputs', input_placeholder)

        output_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size,config.max_phr_len,64),name='output_placeholder')


        f0_input_placeholder= tf.placeholder(tf.float32, shape=(config.batch_size,config.max_phr_len, 1),name='f0_input_placeholder')

        # rand_input_placeholder= tf.placeholder(tf.float32, shape=(config.batch_size,1,1, 4),name='rand_input_placeholder')


        # pho_input_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size,config.max_phr_len, 42),name='pho_input_placeholder')

        prob = tf.placeholder_with_default(1.0, shape=())
        
        phoneme_labels = tf.placeholder(tf.int32, shape=(config.batch_size,config.max_phr_len),name='phoneme_placeholder')
        phone_onehot_labels = tf.one_hot(indices=tf.cast(phoneme_labels, tf.int32), depth=42)

        singer_labels = tf.placeholder(tf.float32, shape=(config.batch_size),name='singer_placeholder')
        singer_onehot_labels = tf.one_hot(indices=tf.cast(singer_labels, tf.int32), depth=12)


        with tf.variable_scope('phone_Model') as scope:
            # regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
            pho_logits, phone_embeding = modules.phone_network(input_placeholder)
            pho_classes = tf.argmax(pho_logits, axis=-1)
            pho_probs = tf.nn.softmax(pho_logits)

        # with tf.variable_scope('Final_Model') as scope:
        #     voc_output = modules.final_net(singer_onehot_labels, f0_input_placeholder, phone_onehot_labels)
        #     voc_output_decoded = tf.nn.sigmoid(voc_output)
        #     scope.reuse_variables()
        #     voc_output_3 = modules.final_net(singer_onehot_labels, f0_input_placeholder, pho_probs)
        #     voc_output_3_decoded = tf.nn.sigmoid(voc_output_3)
            

        with tf.variable_scope('singer_Model') as scope:
            singer_embedding, singer_logits = modules.singer_network(input_placeholder)
            singer_classes = tf.argmax(singer_logits, axis=-1)
            singer_probs = tf.nn.softmax(singer_logits)


        with tf.variable_scope('Generator') as scope: 
            voc_output_2 = modules.GAN_generator(singer_embedding, phone_embeding, f0_input_placeholder)


        saver = tf.train.Saver(max_to_keep= config.max_models_to_keep)


        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess = tf.Session()

        sess.run(init_op)

        ckpt = tf.train.get_checkpoint_state(config.log_dir)

        if ckpt and ckpt.model_checkpoint_path:
            print("Using the model in %s"%ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        # saver.restore(sess, './log/model.ckpt-3999')

        # import pdb;pdb.set_trace()





        voc_file = h5py.File(config.voice_dir+file_name, "r")

        # speaker_file = h5py.File(config.voice_dir+speaker_file, "r")


        feats = np.array(voc_file['feats'])
        # feats = utils.input_to_feats('./54228_chorus.wav_ori_vocals.wav', mode = 1)



        f0 = feats[:,-2]

        # import pdb;pdb.set_trace()

        med = np.median(f0[f0 > 0])

        f0[f0==0] = med

        f0 = f0-12

        feats[:,-2] = feats[:,-2] -12


        f0_nor = (f0 - min_feat[-2])/(max_feat[-2]-min_feat[-2])

        feats = (feats-min_feat)/(max_feat-min_feat)

        stft = np.array(voc_file['voc_stft'])



       # pho_target = np.array(voc_file["phonemes"])



        in_batches_f0, nchunks_in = utils.generate_overlapadd(f0_nor.reshape(-1,1))

       # in_batches_pho, nchunks_in_pho = utils.generate_overlapadd(pho_target.reshape(-1,1))

        in_batches_feat, kaka = utils.generate_overlapadd(feats)

        in_batches_stft, kaka = utils.generate_overlapadd(stft)

        # import pdb;pdb.set_trace()




        out_batches_feats = []


        out_batches_feats_1 = []

        out_batches_feats_gan = []




        for in_batch_f0, in_batch_stft  in zip(in_batches_f0, in_batches_stft):

            in_batch_f0= in_batch_f0.reshape([config.batch_size, config.max_phr_len, 1])

           # in_batch_pho_target = in_batch_pho_target.reshape([config.batch_size, config.max_phr_len])

            # in_batch_pho_target = sess.run(pho_probs, feed_dict = {input_placeholder: in_batch_feat})

            output_feats_gan = sess.run(voc_output_2, feed_dict = {input_placeholder: in_batch_stft,
              f0_input_placeholder: in_batch_f0, singer_labels: np.ones(30)*singer_index})



            out_batches_feats_gan.append(output_feats_gan /2 +0.5)



            # out_batches_voc_stft_phase.append(output_voc_stft_phase)



        # import pdb;pdb.set_trace()



        out_batches_feats_gan = np.array(out_batches_feats_gan)
        # import pdb;pdb.set_trace()
        out_batches_feats_gan = utils.overlapadd(out_batches_feats_gan, nchunks_in) 


        feats = feats *(max_feat-min_feat)+min_feat



        out_batches_feats_gan = out_batches_feats_gan * (max_feat[:-2]-min_feat[:-2])+min_feat[:-2]



        out_batches_feats_gan= out_batches_feats_gan[:len(feats)]

        # first_op = np.concatenate([out_batches_feats,feats[:,-2:]], axis = -1)

        # pho_op = np.concatenate([out_batches_feats_1,feats[:,-2:]], axis = -1)

        gan_op = np.concatenate([out_batches_feats_gan,feats[:,-2:]], axis = -1)


        # # import pdb;pdb.set_trace()
        gan_op = np.ascontiguousarray(gan_op)

        # import pdb;pdb.set_trace()

        # pho_op = np.ascontiguousarray(pho_op)

        # first_op = np.ascontiguousarray(first_op)

        if show_plots:

            plt.figure(1)

            ax1 = plt.subplot(211)

            plt.imshow(feats[:,:60].T,aspect='auto',origin='lower')

            ax1.set_title("Ground Truth Vocoder Features", fontsize=10)

            # ax2 = plt.subplot(312, sharex = ax1, sharey = ax1)

            # plt.imshow(out_batches_feats[:,:60].T,aspect='auto',origin='lower')

            # ax2.set_title("Cross Entropy Output Vocoder Features", fontsize=10)

            ax3 =plt.subplot(212, sharex = ax1, sharey = ax1)

            ax3.set_title("GAN Vocoder Output Features", fontsize=10)

            # plt.imshow(out_batches_feats_1[:,:60].T,aspect='auto',origin='lower')
            #
            # plt.subplot(414, sharex = ax1, sharey = ax1)

            plt.imshow(out_batches_feats_gan[:,:60].T,aspect='auto',origin='lower')

            # plt.figure(2)

            # plt.subplot(211)

            # plt.imshow(feats[:,60:-2].T,aspect='auto',origin='lower')

            # plt.subplot(212)

            # plt.imshow(out_batches_feats[:,-4:].T,aspect='auto',origin='lower')

            plt.show()

            save_file = input("Which files to synthesise G for GAN, B for Binary Entropy, "
                              "O for original, or any combination. Default is None").upper() or "N"

        else:
            save_file = input("Which files to synthesise G for GAN, B for Binary Entropy, "
                          "O for original, or any combination. Default is all (GBO)").upper() or "GBO"

        if "G" in save_file:

            utils.feats_to_audio(gan_op[:,:],file_name[:-4]+'gan_op.wav')

            print("GAN file saved to {}".format(os.path.join(config.val_dir,file_name[:-4]+'gan_op.wav' )))

        if "O" in save_file:

            utils.feats_to_audio(feats[:, :], file_name[:-4]+'ori_op.wav')

            print("Originl file, resynthesized via WORLD vocoder saved to {}".format(os.path.join(config.val_dir, file_name[:-4] + 'ori_op.wav')))
            #
        if "B" in save_file:
            # # utils.feats_to_audio(pho_op[:5000,:],file_name[:-4]+'phoop.wav')
            #
            utils.feats_to_audio(first_op[:,:],file_name[:-4]+'bce_op.wav')
            print("Binar cross entropy file saved to {}".format(os.path.join(config.val_dir, file_name[:-4] + 'bce_op.wav')))


        # utils.query_yes_no("Anything Else or Exit?")
        #
        # import pdb;pdb.set_trace()






if __name__ == '__main__':
    if '-l' in sys.argv:
        index_log = sys.argv.index('-l')
        import pdb;pdb.set_trace()
    if sys.argv[1] == '-train' or sys.argv[1] == '--train' or sys.argv[1] == '--t' or sys.argv[1] == '-t':
        print("Training")
        tf.app.run(main=train)
    elif sys.argv[1] == '-synth' or sys.argv[1] == '--synth' or sys.argv[1] == '--s' or sys.argv[1] == '-s':
        if len(sys.argv) < 3:
            print("Please give a file to synthesize")
        else:
            file_name = sys.argv[2]
            if not file_name.endswith('.hdf5'):
                file_name = file_name + '.hdf5'
            if not file_name in os.listdir(config.voice_dir):
                print("Currently only supporting hdf5 files which are in the dataset, will be expanded later.")
            FLAG_PLOT = utils.query_yes_no("Plot plots?", default="yes")
            # import pdb;pdb.set_trace()

            if len(sys.argv) < 4:
               # singer_name = file_name.split('_')[1]
               # print("Synthesizing with same singer as input file, {}, to change, please give a different singer after the song file".format(singer_name))
               # singer_index = config.singers.index(singer_name)
                synth_file(file_name, show_plots = FLAG_PLOT)

            else:
                singer_name = sys.argv[3]
                if singer_name not in config.singers:
                    print("Please give a valid singer name to synthesize")
                    print("Valid names are:")
                    for singer in config.singers:
                        print(singer)
                else:
                    singer_index = config.singers.index(singer_name)
                    synth_file(file_name, singer_index, show_plots = FLAG_PLOT)






