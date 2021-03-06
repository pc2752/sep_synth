from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import tensorflow as tf
import numpy as np
from tensorflow.python import debug as tf_debug
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib import rnn
import config


tf.logging.set_verbosity(tf.logging.INFO)




def wavenet_block(inputs, conditioning, is_train, dilation_rate = 2, kernel_size = config.kernel_size, name = "name"):

    pad = (kernel_size - 1) * dilation_rate

    conditioning = tf.layers.batch_normalization(tf.layers.conv1d(conditioning, config.num_filters, 1, dilation_rate = 1, padding = 'valid', name = name+"_cond"), training = is_train)

    con_pad_forward = tf.pad(inputs, [[0,0],[dilation_rate,0],[0,0]],"CONSTANT")

    con_sig_forward = tf.layers.batch_normalization(tf.layers.conv1d(con_pad_forward, config.num_filters, kernel_size, dilation_rate = dilation_rate, padding = 'valid', name = name+"_1"), training = is_train)

    sig = tf.sigmoid(con_sig_forward + conditioning)

    con_tanh_forward = tf.layers.batch_normalization(tf.layers.conv1d(con_pad_forward, config.num_filters, kernel_size, dilation_rate = dilation_rate, padding = 'valid', name = name+"_2"), training = is_train)

    tanh = tf.tanh(con_tanh_forward + conditioning)

    outputs = tf.multiply(sig,tanh)

    residual = outputs + inputs

    skip = tf.layers.conv1d(outputs,config.skip_filters,1, name = name+"_skip")

    residual = tf.layers.conv1d(residual,config.num_filters,1, name = name+"_residual")
    
    return skip, residual

def wave_archi(inputs, conditioning, is_train):


    receptive_field = 2**config.wavenet_layers

    inputs = tf.pad(inputs, [[0,0],[config.first_conv -1 ,0],[0,0]],"CONSTANT")

    residual = tf.layers.batch_normalization(tf.layers.conv1d(inputs, config.num_filters, config.first_conv, name = "first_conv"), training = is_train)

    skips = []

    output = tf.layers.conv1d(residual,config.skip_filters,1, name = "first_skip")

    for i in range(config.wavenet_layers):
        skip, residual = wavenet_block(residual,conditioning, is_train, dilation_rate = config.dilation_rates[i], name = "npss_block_"+str(i+1))
        skips.append(skip)
    for skip in skips:
        output+=skip

    conditioning = tf.layers.batch_normalization(tf.layers.conv1d(conditioning, config.skip_filters, 1, dilation_rate = 1, padding = 'valid', name = "cond"), training = is_train)

    output = output + conditioning

    output = tf.nn.tanh(output)

    output = tf.layers.conv1d(output,config.num_phos,1, name = "Output" )

    return output



def deconv2d(input_, output_shape,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="deconv2d"):
  with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.random_normal_initializer(stddev=stddev))
    
    # try:
    deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1], name = name)

    biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    # if with_w:
    #   return deconv, w, biases
    # else:
  return deconv
def selu(x):
   alpha = 1.6732632423543772848170429916717
   scale = 1.0507009873554804934193349852946
   return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))


def bi_dynamic_stacked_RNN(x, input_lengths, scope='RNN'):
    with tf.variable_scope(scope):
    # x = tf.layers.dense(x, 128)

        cell = tf.nn.rnn_cell.LSTMCell(num_units=config.lstm_size, state_is_tuple=True)
        cell2 = tf.nn.rnn_cell.LSTMCell(num_units=config.lstm_size, state_is_tuple=True)

        outputs, _state1, state2  = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            cells_fw=[cell,cell2],
            cells_bw=[cell,cell2],
            dtype=config.dtype,
            sequence_length=input_lengths,
            inputs=x)

    return outputs

def bi_static_stacked_RNN(x, scope='RNN'):
    """
    Input and output in batch major format
    """
    with tf.variable_scope(scope):
        x = tf.unstack(x, config.max_phr_len, 1)

        output = x
        num_layer = 2
        # for n in range(num_layer):
        lstm_fw = tf.nn.rnn_cell.LSTMCell(config.lstm_size, state_is_tuple=True)
        lstm_bw = tf.nn.rnn_cell.LSTMCell(config.lstm_size, state_is_tuple=True)

        _initial_state_fw = lstm_fw.zero_state(config.batch_size, tf.float32)
        _initial_state_bw = lstm_bw.zero_state(config.batch_size, tf.float32)

        output, _state1, _state2 = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw, lstm_bw, output, 
                                                  initial_state_fw=_initial_state_fw,
                                                  initial_state_bw=_initial_state_bw, 
                                                  scope='BLSTM_')
        output = tf.stack(output)
        output_fw = output[0]
        output_bw = output[1]
        output = tf.transpose(output, [1,0,2])


        # output = tf.layers.dense(output, config.output_features, activation=tf.nn.relu) # Remove this to use cbhg

        return output




def bi_dynamic_RNN(x, input_lengths, scope='RNN'):
    """
    Stacked dynamic RNN, does not need unpacking, but needs input_lengths to be specified
    """

    with tf.variable_scope(scope):

        cell = tf.nn.rnn_cell.LSTMCell(num_units=config.lstm_size, state_is_tuple=True)

        outputs, states  = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell,
            cell_bw=cell,
            dtype=config.dtype,
            sequence_length=input_lengths,
            inputs=x)

        outputs = tf.concat(outputs, axis=2)

    return outputs


def RNN(x, scope='RNN'):
    with tf.variable_scope(scope):
        x = tf.unstack(x, config.max_phr_len, 1)

        lstm_cell = rnn.BasicLSTMCell(num_units=config.lstm_size)

        # Get lstm cell output
        outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=config.dtype)
        outputs=tf.stack(outputs)
        outputs = tf.transpose(outputs, [1,0,2])

    return outputs


def highwaynet(inputs, scope='highway', units=config.highway_units):
    with tf.variable_scope(scope):
        H = tf.layers.dense(
        inputs,
        units=units,
        activation=tf.nn.relu,
        name='H')
        T = tf.layers.dense(
        inputs,
        units=units,
        activation=tf.nn.sigmoid,
        name='T',
        bias_initializer=tf.constant_initializer(-1.0))
    return H * T + inputs * (1.0 - T)


def conv(inputs, kernel_size, filters=config.conv_filters, activation=config.conv_activation, training=True, scope='conv'):
  with tf.variable_scope(scope):
    x = tf.layers.conv1d(inputs,filters=filters,kernel_size=kernel_size,activation=activation,padding='same')
    return tf.layers.batch_normalization(x, training=training)


# def build_encoder(inputs):
#     embedding_encoder = variable_scope.get_variable("embedding_encoder", [config.vocab_size, config.inp_embedding_size], dtype=config.dtype)

def conv_bank(inputs, scope='conv_bank', num_layers=config.num_conv_layers, training=True):
    with tf.variable_scope(scope):
        outputs = [conv(inputs, k, training=training, scope='conv_%d' % k) for k in range(1, num_layers+1)]
        outputs = tf.concat(outputs, axis=-1)
        outputs = tf.layers.max_pooling1d(outputs,pool_size=2,strides=1,padding='same')
    return outputs




        




def nr_wavenet_block_d(conditioning,filters=2, scope = 'nr_wavenet_block', name = "name"):

    with tf.variable_scope(scope):
    # inputs = tf.reshape(inputs, [config.batch_size, config.max_phr_len, config.input_features])

        # con_pad_forward = tf.pad(conditioning, [[0,0],[dilation_rate,0],[0,0]],"CONSTANT")
        # con_pad_backward = tf.pad(conditioning, [[0,0],[0,dilation_rate],[0,0]],"CONSTANT")
        con_sig_forward = tf.layers.conv1d(conditioning, config.wavenet_filters, filters, padding = 'valid', name = name+"1")
        # con_sig_backward = tf.layers.conv1d(con_pad_backward, config.wavenet_filters, 2, dilation_rate = dilation_rate, padding = 'valid', name = name+"2")
        # con_sig = tf.layers.conv1d(conditioning,config.wavenet_filters,1)

        sig = tf.sigmoid(con_sig_forward)


        con_tanh_forward = tf.layers.conv1d(conditioning, config.wavenet_filters, filters, padding = 'valid', name = name+"2")

        tanh = tf.tanh(con_tanh_forward)


        outputs = tf.multiply(sig,tanh)

        skip = tf.layers.conv1d(outputs,config.wavenet_filters,1, name = name+"5")

        residual = skip

    return skip, residual


def nr_wavenet(inputs, num_block = config.wavenet_layers):
    prenet_out = tf.layers.dense(inputs, config.lstm_size*2)
    prenet_out = tf.layers.dense(prenet_out, config.lstm_size)

    receptive_field = 2**num_block

    first_conv = tf.layers.conv1d(prenet_out, config.wavenet_filters, 1)
    skips = []
    skip, residual = nr_wavenet_block(first_conv, dilation_rate=1, scope = "nr_wavenet_block_0")
    output = skip
    for i in range(num_block):
        skip, residual = nr_wavenet_block(residual, dilation_rate=2**(i+1), scope = "nr_wavenet_block_"+str(i+1))
        skips.append(skip)
    for skip in skips:
        output+=skip
    output = output+first_conv

    output = tf.nn.relu(output)

    output = tf.layers.conv1d(output,config.wavenet_filters,1)

    output = tf.nn.relu(output)

    output = tf.layers.conv1d(output,config.wavenet_filters,1)

    output = tf.nn.relu(output)

    harm = tf.layers.dense(output, 60, activation=tf.nn.relu)
    ap = tf.layers.dense(output, 4, activation=tf.nn.relu)
    f0 = tf.layers.dense(output, 64, activation=tf.nn.relu) 
    f0 = tf.layers.dense(f0, 1, activation=tf.nn.relu)
    vuv = tf.layers.dense(ap, 1, activation=tf.nn.sigmoid)

    return harm, ap, f0, vuv



def nr_wavenet_block(inputs, is_train, dilation_rate = 2, name = "name"):

    con_pad_forward = tf.pad(inputs, [[0,0],[dilation_rate,dilation_rate],[0,0]],"CONSTANT")

    con_sig_forward = tf.layers.batch_normalization(tf.layers.conv1d(con_pad_forward, config.filters, 3, dilation_rate = dilation_rate, padding = 'valid', name = name+"_1", kernel_initializer=tf.random_normal_initializer(stddev=0.02)), training = is_train, name = name+"_1BN")

    sig = tf.sigmoid(con_sig_forward)

    con_tanh_forward = tf.layers.batch_normalization(tf.layers.conv1d(con_pad_forward, config.filters, 3, dilation_rate = dilation_rate, padding = 'valid', name = name+"_2", kernel_initializer=tf.random_normal_initializer(stddev=0.02)), training = is_train, name = name+"_2BN")

    tanh = tf.tanh(con_tanh_forward)

    outputs = tf.multiply(sig,tanh)

    skip = tf.layers.batch_normalization(tf.layers.conv1d(outputs,config.filters,1, name = name+"_3", kernel_initializer=tf.random_normal_initializer(stddev=0.02)), training = is_train, name = name+"_3BN")

    residual = skip + inputs

    return skip, residual


# def wave_archi(inputs, is_train):

#     inputs = tf.squeeze(inputs)

#     prenet_out = tf.layers.dense(inputs, config.filters, name = "Phone1")

#     receptive_field = 2**config.wavenet_layers

#     first_conv = tf.layers.batch_normalization(tf.layers.conv1d(prenet_out, config.filters, 1, name = "Phone2", kernel_initializer=tf.random_normal_initializer(stddev=0.02)), training = is_train, name = "Phone2BN")
#     skips = []
#     skip, residual = nr_wavenet_block(first_conv, is_train, dilation_rate = 1, name = "Phone_block_0")
#     output = skip
#     for i in range(config.wavenet_layers):
#         skip, residual = nr_wavenet_block(residual, is_train, dilation_rate = 2**(i+1), name = "Phone_block_"+str(i+1))
#         skips.append(skip)
#     for skip in skips:
#         output+=skip
#     output = output+first_conv

#     output = tf.nn.relu(output)

#     output = tf.layers.batch_normalization(tf.layers.conv1d(output,config.filters,1, name = "P_F_1" , kernel_initializer=tf.random_normal_initializer(stddev=0.02)), training = is_train, name = "P_F_1BN")

#     output = tf.nn.relu(output)

#     return tf.reshape(output, [config.batch_size, config.max_phr_len, 1, -1])



def encoder_conv_block_gan(inputs, layer_num, is_train, num_filters = config.filters):

    output = tf.layers.batch_normalization(tf.nn.relu(tf.layers.conv2d(inputs, num_filters * 2**int(layer_num/2), (config.filter_len,1)
        , strides=(2,1),  padding = 'same', name = "G_"+str(layer_num), kernel_initializer=tf.random_normal_initializer(stddev=0.02))), training = is_train, name = "GBN_"+str(layer_num))
    return output

def decoder_conv_block_gan(inputs, layer, layer_num, is_train, num_filters = config.filters):

    deconv = tf.image.resize_images(inputs, size=(int(config.max_phr_len/2**(config.encoder_layers - 1 - layer_num)),1), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # embedding = tf.tile(embedding,[1,int(config.max_phr_len/2**(config.encoder_layers - 1 - layer_num)),1,1])

    deconv = tf.layers.batch_normalization( tf.nn.relu(tf.layers.conv2d(deconv, layer.shape[-1]
        , (config.filter_len,1), strides=(1,1),  padding = 'same', name =  "D_"+str(layer_num), kernel_initializer=tf.random_normal_initializer(stddev=0.02))), training = is_train, name =  "DBN_"+str(layer_num))

    # embedding =tf.nn.relu(tf.layers.conv2d(embedding, layer.shape[-1]
    #     , (config.filter_len,1), strides=(1,1),  padding = 'same', name =  "DEnc_"+str(layer_num)))

    deconv =  tf.concat([deconv, layer], axis = -1)

    return deconv

def encoder_conv_block(inputs, layer_num, is_train, num_filters = config.filters):

    output = tf.layers.batch_normalization(tf.nn.relu(tf.layers.conv2d(inputs, num_filters * 2**int(layer_num/2), (config.filter_len,1)
        , strides=(2,1),  padding = 'same', name = "G_"+str(layer_num))), training = is_train, name = "GBN_"+str(layer_num))
    return output

def decoder_conv_block(inputs, layer, layer_num, is_train, num_filters = config.filters):

    deconv = tf.image.resize_images(inputs, size=(int(config.max_phr_len/2**(config.encoder_layers - 1 - layer_num)),1), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # embedding = tf.tile(embedding,[1,int(config.max_phr_len/2**(config.encoder_layers - 1 - layer_num)),1,1])

    deconv = tf.layers.batch_normalization( tf.nn.relu(tf.layers.conv2d(deconv, layer.shape[-1]
        , (config.filter_len,1), strides=(1,1),  padding = 'same', name =  "D_"+str(layer_num))), training = is_train, name =  "DBN_"+str(layer_num))

    # embedding =tf.nn.relu(tf.layers.conv2d(embedding, layer.shape[-1]
    #     , (config.filter_len,1), strides=(1,1),  padding = 'same', name =  "DEnc_"+str(layer_num)))

    deconv =  tf.concat([deconv, layer], axis = -1)

    return deconv

def encoder_decoder_archi(inputs, is_train):
    """
    Input is assumed to be a 4-D Tensor, with [batch_size, phrase_len, 1, features]
    """

    encoder_layers = []

    encoded = inputs

    encoder_layers.append(encoded)

    for i in range(config.encoder_layers):
        encoded = encoder_conv_block(encoded, i, is_train)
        encoder_layers.append(encoded)
    
    encoder_layers.reverse()



    decoded = encoder_layers[0]

    for i in range(config.encoder_layers):
        decoded = decoder_conv_block(decoded, encoder_layers[i+1], i, is_train)

    return decoded

def encoder_decoder_archi_gan(inputs, is_train):
    """
    Input is assumed to be a 4-D Tensor, with [batch_size, phrase_len, 1, features]
    """

    encoder_layers = []

    encoded = inputs

    encoder_layers.append(encoded)

    for i in range(config.encoder_layers):
        encoded = encoder_conv_block_gan(encoded, i, is_train)
        encoder_layers.append(encoded)
    
    encoder_layers.reverse()



    decoded = encoder_layers[0]

    for i in range(config.encoder_layers):
        decoded = decoder_conv_block_gan(decoded, encoder_layers[i+1], i, is_train)

    return decoded


def phone_network(inputs, is_train):

    # inputs = tf.reshape(inputs, [config.batch_size, config.max_phr_len, 1, -1])

    inputs = tf.layers.batch_normalization(tf.layers.dense(inputs, config.wavenet_filters
        , name = "P_in"), training = is_train)
 
    output = wave_archi(inputs, is_train)


    output = tf.layers.batch_normalization(tf.layers.dense(output, config.num_phos, name = "P_F"), training = is_train)

    output = tf.squeeze(output)

    return output

def f0_network(inputs, is_train):

    inputs = tf.reshape(inputs, [config.batch_size, config.max_phr_len, 1, -1])

    inputs = tf.layers.batch_normalization(tf.layers.dense(inputs, config.wavenet_filters
        , name = "F_in"), training = is_train)


    output = wave_archi(inputs, is_train)


    output = tf.layers.batch_normalization(tf.layers.dense(output, config.num_f0, name = "F_F"), training = is_train)

    output = tf.squeeze(output)


    return output

def singer_network(inputs, is_train):

    inputs = tf.reshape(inputs, [config.batch_size, config.max_phr_len, 1, -1])

    inputs = tf.layers.batch_normalization(tf.layers.dense(inputs, config.wavenet_filters
        , name = "S_in"), training = is_train)

    encoded = inputs

    for i in range(config.encoder_layers):
        encoded = encoder_conv_block(encoded, i, is_train)
    encoded = tf.squeeze(encoded)
    output = tf.layers.batch_normalization(tf.layers.dense(encoded, config.num_singers, name = "S_F"), training = is_train)
    return encoded, output

# def full_network(content_embedding, singer_embedding, f0_embedding, is_train):

#     inputs = tf.concat([content_embedding, singer_embedding, f0_embedding], axis = -1)
#     inputs = tf.reshape(inputs, [config.batch_size, 1, 1, -1])

#     inputs = tf.layers.batch_normalization(tf.layers.dense(inputs, config.wavenet_filters*4
#         , name = "S_in"), training = is_train)

#     decoded = inputs

#     for i in range(config.encoder_layers):
#         decoded = decoder_conv_block_full(decoded, i, is_train)

#     output = tf.layers.batch_normalization(tf.layers.dense(decoded, config.output_features, name = "Fu_F"), training = is_train)

#     return tf.squeeze(output)


def full_network(inputs, f0, phos,  singer_label, is_train):

    inputs = tf.layers.batch_normalization(tf.layers.dense(inputs, config.filters
        , name = "I_in", kernel_initializer=tf.random_normal_initializer(stddev=0.02)), training = is_train)

    singer_label = tf.tile(tf.reshape(singer_label,[config.batch_size,1,-1]),[1,config.max_phr_len,1])

    inputs = tf.concat([inputs, f0, phos,singer_label], axis = -1)

    inputs = tf.reshape(inputs, [config.batch_size, config.max_phr_len , 1, -1])

    inputs = tf.layers.batch_normalization(tf.layers.dense(inputs, config.filters
        , name = "S_in", kernel_initializer=tf.random_normal_initializer(stddev=0.02)), training = is_train)


    output = encoder_decoder_archi_gan(inputs, is_train)


    output = tf.tanh(tf.layers.batch_normalization(tf.layers.dense(output, config.output_features, name = "Fu_F", kernel_initializer=tf.random_normal_initializer(stddev=0.02)), training = is_train, name = "bn_fu_out"))

    return tf.squeeze(output)

def discriminator(inputs, phos, f0, singer_label, is_train):

    singer_label = tf.tile(tf.reshape(singer_label,[config.batch_size,1,-1]),[1,config.max_phr_len,1])

    inputs = tf.concat([inputs, f0, phos,singer_label], axis = -1)

    inputs = tf.reshape(inputs, [config.batch_size, config.max_phr_len , 1, -1])



    inputs = tf.layers.batch_normalization(tf.layers.dense(inputs, config.filters *2 
        , name = "S_in", kernel_initializer=tf.random_normal_initializer(stddev=0.02)), training = is_train, name = "bn_fu_1")

    # encoded = inputs

    encoded = inputs

    for i in range(config.encoder_layers):
        encoded = encoder_conv_block_gan(encoded, i, is_train)
    encoded = tf.squeeze(encoded)
    # output = tf.layers.batch_normalization(tf.layers.dense(encoded, config.num_singers, name = "S_F"), training = is_train, name = "bn_dis")

    # output = wave_archi(inputs, is_train)

    output = tf.layers.batch_normalization(tf.layers.dense(encoded, 1, name = "Fu_F", kernel_initializer=tf.random_normal_initializer(stddev=0.02)), training = is_train, name = "bn_fu_out")

    return tf.squeeze(output)
    # tf.squeeze(output[:,int(config.max_phr_len/2)-2:int(config.max_phr_len/2)+1, :])

def main():    
    vec = tf.placeholder("float", [config.batch_size, config.max_phr_len, config.input_features])
    tec = np.random.rand(config.batch_size, config.max_phr_len,config.input_features) #  batch_size, time_steps, features
    is_train = tf.placeholder(tf.bool, name="is_train")
    # seqlen = tf.placeholder("float", [config.batch_size, 256])
    # with tf.variable_scope('singer_Model') as scope:
    #     singer_emb, outs_sing = singer_network(vec, is_train)
    # with tf.variable_scope('f0_Model') as scope:
    #     outs_f0 = f0_network(vec, is_train)
    # with tf.variable_scope('phone_Model') as scope:
    #     outs_pho = phone_network(vec, is_train)
    with tf.variable_scope('full_Model') as scope:
        out_put = discriminator(vec,is_train)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    op= sess.run(out_put, feed_dict={vec: tec, is_train: True})
    # writer = tf.summary.FileWriter('.')
    # writer.add_graph(tf.get_default_graph())
    # writer.add_summary(summary, global_step=1)
    import pdb;pdb.set_trace()


if __name__ == '__main__':
  main()