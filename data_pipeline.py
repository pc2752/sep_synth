import numpy as np
import os
import time
import h5py

import matplotlib.pyplot as plt
import collections
import config
import utils
import soundfile as sf

from scipy.ndimage import filters


def one_hotize(inp, max_index=config.num_phos):


    output = np.eye(max_index)[inp.astype(int)]

    return output



def gen_train_val():
    mix_list = [x for x in os.listdir(config.backing_dir) if x.endswith('.hdf5') and x.startswith('med') ]

    train_list = mix_list[:int(len(mix_list)*config.split)]

    val_list = mix_list[int(len(mix_list)*config.split):]

    utils.list_to_file(val_list,config.log_dir+'val_files.txt')

    utils.list_to_file(train_list,config.log_dir+'train_files.txt')


def data_gen_pho(mode = 'Train', sec_mode = 0):

    mix_list = [x for x in os.listdir(config.voice_dir) if x.endswith('.hdf5') and x.startswith('nus') and x.split('_')[1] not in ['ADIZ', 'JLEE', 'JTAN', 'KENN']]

    val_list = [x for x in os.listdir(config.voice_dir) if x.endswith('.hdf5')and x.startswith('nus') and x.split('_')[1] in ['ADIZ', 'JLEE', 'JTAN', 'KENN'] and not x.startswith('nus_KENN_read')]

    stat_file = h5py.File(config.stat_dir+'stats.hdf5', mode='r')

    max_feat = np.array(stat_file["feats_maximus"])
    min_feat = np.array(stat_file["feats_minimus"])
    max_voc = np.array(stat_file["voc_stft_maximus"])
    min_voc = np.array(stat_file["voc_stft_minimus"])
    max_back = np.array(stat_file["back_stft_maximus"])
    min_back = np.array(stat_file["back_stft_minimus"])
    max_mix = np.array(max_voc)+np.array(max_back)
    stat_file.close()


    max_files_to_process = int(config.batch_size/config.samples_per_file)

    if mode == "Train":
        num_batches = config.batches_per_epoch_train
        file_list = mix_list

    else: 
        num_batches = config.batches_per_epoch_val
        file_list = val_list



    for k in range(num_batches):
        pho_targs = []

        mix_in = []
        for i in range(max_files_to_process):


            voc_index = np.random.randint(0,len(file_list))
            voc_to_open = file_list[voc_index]




            voc_file = h5py.File(config.voice_dir+voc_to_open, "r")


            voc_stft = np.array(voc_file['voc_stft'])


            if voc_to_open.startswith('nus'):
                if not  "phonemes" in voc_file:
                    print(voc_file)
                    Flag = False
                else: 
                    Flag = True
                    pho_target = np.array(voc_file["phonemes"])
                    pho_target = [config.phonemas_all.index(config.phonemas[x]) for x in pho_target]
                    # singer_name = voc_to_open.split('_')[1]
                    # singer_index = config.singers.index(singer_name)
            else:
                Flag = False




            for j in range(config.samples_per_file):
                    voc_idx = np.random.randint(0,len(voc_stft)-config.max_phr_len)
                    # bac_idx = np.random.randint(0,len(back_stft)-config.max_phr_len)
                    mix_stft = voc_stft[voc_idx:voc_idx+config.max_phr_len,:]
                    # *np.clip(np.random.rand(1),0.5,0.9) + back_stft[bac_idx:bac_idx+config.max_phr_len,:]*np.clip(np.random.rand(1),0.0,0.9) + np.random.rand(config.max_phr_len,config.input_features)*np.clip(np.random.rand(1),0.0,config.noise_threshold)
                    mix_in.append(mix_stft)

                    if Flag:
                        pho_targs.append(pho_target[voc_idx:voc_idx+config.max_phr_len])

            mix_in = (np.array(mix_in) - min_voc)/(max_voc - min_voc)

            assert mix_in.max()<=1.0 and mix_in.min()>=0

            yield mix_in, pho_targs



def data_gen_full(mode = 'Train', sec_mode = 0):

    casas_list = [x for x in os.listdir(config.voice_dir) if x.endswith('.hdf5') and x.startswith('casas') and not x in config.do_not_use]

    nus_list = [x for x in os.listdir(config.voice_dir) if x.endswith('.hdf5') and x.startswith('nus') and x.split('_')[3] not in ['15.hdf5','20.hdf5'] and not x.startswith('nus_KENN_read') ]

    pho_list = nus_list 
    # + casas_list[:int(0.8*len(casas_list))]

    mix_list_med = [x for x in os.listdir(config.voice_dir) if x.endswith('.hdf5') and x.startswith('med') and not x.split('_')[1] in ['MusicDelta', 'ClaraBerryAndWooldog','ClaraBerryAndWooldog','CelestialShore', 'Schumann', 'Mozart', 'NightPanther', 'Debussy', 'HeladoNegro']]

    mix_list = pho_list + mix_list_med[:int(0.8*len(mix_list_med))]

    back_list = [x for x in os.listdir(config.backing_dir) if x.endswith('.hdf5') and not x.startswith('._') and not x.startswith('mir') and not x.startswith('med')]

    val_list = [x for x in os.listdir(config.voice_dir) if x.endswith('.hdf5')and x.startswith('nus') and x.split('_')[3] in ['15.hdf5','20.hdf5']  and not x.startswith('nus_KENN_read')] 
    # + casas_list[int(0.8)*len(casas_list):]
    # + mix_list_med[int(0.8*len(mix_list_med)):]

    # import pdb;pdb.set_trace()


    stat_file = h5py.File(config.stat_dir+'stats.hdf5', mode='r')

    max_feat = np.array(stat_file["feats_maximus"])
    min_feat = np.array(stat_file["feats_minimus"])
    max_voc = np.array(stat_file["voc_stft_maximus"])
    min_voc = np.array(stat_file["voc_stft_minimus"])
    max_back = np.array(stat_file["back_stft_maximus"])
    min_back = np.array(stat_file["back_stft_minimus"])
    max_mix = np.array(max_voc)+np.array(max_back)
    stat_file.close()


    max_files_to_process = int(config.batch_size/config.samples_per_file)

    if mode == "Train":
        num_batches = config.batches_per_epoch_train
        file_list = pho_list

    else: 
        num_batches = config.batches_per_epoch_val
        file_list = val_list

    for k in range(num_batches):
        pho_targs = []

        # f0_targs = []

        singer_targs = []

        mix_in = []

        voc_out = []

        f0_out = []

        for i in range(max_files_to_process):


            voc_index = np.random.randint(0,len(file_list))
            voc_to_open = file_list[voc_index]


            voc_file = h5py.File(config.voice_dir+voc_to_open, "r")


            voc_stft = np.array(voc_file['voc_stft'])

            # voc_stft = voc_stft/voc_stft.max()

            singer_name = voc_to_open.split('_')[1]
            singer_index = config.singers.index(singer_name)
            feats = np.array(voc_file['feats'])

            if np.isnan(feats).any():
                print("nan found")
                import pdb;pdb.set_trace()

            f0 = feats[:,-2]

            med = np.median(f0[f0 > 0])

            f0[f0==0] = med

            f0_nor = (f0 - min_feat[-2])/(max_feat[-2]-min_feat[-2])


            f0_quant = np.rint(f0_nor*config.num_f0) + 1

            f0_quant = f0_quant * (1-feats[:,-1]) 

            back_index = np.random.randint(0,len(back_list))

            back_to_open = back_list[back_index]

            back_file = h5py.File(config.backing_dir+back_to_open, "r")

            back_stft = np.array(back_file['back_stft'])

            pho_target = voc_file["phonemes"]


            # if voc_to_open.startswith('nus'):
            #     if not  "phonemes" in voc_file:
            #         print(voc_file)
            #         Flag = False
            #     else: 
            #         Flag = True
            #         pho_target = voc_file["phonemes"]
            #         # import pdb;pdb.set_trace()
            #         pho_target = [config.phonemas_all.index(config.phonemas[int(x)]) for x in pho_target]

            # elif voc_to_open.startswith('casas'):
            #     if not  "phonemes" in voc_file:
            #         print(voc_file)
            #         Flag = False
            #     else: 
            #         Flag = True
            #         pho_target = voc_file["phonemes"]
            #         # import pdb;pdb.set_trace()
            #         pho_target = [config.phonemas_all.index(config.phonemas_full[int(x)]) for x in pho_target[:,1]]


            for j in range(config.samples_per_file):
                    voc_idx = np.random.randint(0,len(voc_stft)-config.max_phr_len)
                    bac_idx = np.random.randint(0,len(back_stft)-config.max_phr_len)

                    mix_degree = np.clip(np.random.rand(1),0.0,0.9) 

                    mix_stft = (voc_stft[voc_idx:voc_idx+config.max_phr_len,:] + back_stft[bac_idx:bac_idx+config.max_phr_len,:] * mix_degree) / (1+mix_degree)

                    mix_in.append(mix_stft)

                    pho_t = pho_target[voc_idx:voc_idx+config.max_phr_len]

                    # if voc_to_open.startswith('nus'):

                    #     pho_t = [config.phonemas_all.index(config.phonemas[int(x)]) for x in pho_t]
                        # import pdb;pdb.set_trace()
                    # elif voc_to_open.startswith('casas'):
                    #     pho_t = [config.phonemas_all.index(config.phonemas_full[int(x)]) for x in pho_t[:,1]]
                    #     # import pdb;pdb.set_trace()


                    pho_targs.append(pho_t)

                    voc_out.append(feats[voc_idx:voc_idx+config.max_phr_len,:])

                    f0_out.append(f0_quant[voc_idx:voc_idx+config.max_phr_len])

                    singer_targs.append(singer_index)


        # mix_in = (np.array(mix_in) - min_voc)/(max_voc - min_voc)

        mix_in = np.clip(np.array(mix_in), 0.0, 1.0)



        pho_targs = np.array(pho_targs)

        voc_out = (np.array(voc_out) - min_feat)/(max_feat - min_feat)

        voc_out = voc_out[:,:,:-2]

        f0_out = np.array(f0_out)

        singer_targs = np.array(singer_targs)

        # assert mix_in.max()<=1.0 and mix_in.min()>=0

        # assert voc_out.max()<=1.0 and voc_out.min()>=0

        yield mix_in, singer_targs, voc_out, f0_out, pho_targs


def get_stats():
    voc_list = [x for x in os.listdir(config.voice_dir) if x.endswith('.hdf5') and x.startswith('nus') or x.startswith('med') or x.startswith('casas') and not x.startswith('nus_KENN_read') ]

    back_list = [x for x in os.listdir(config.backing_dir) if x.endswith('.hdf5') and not x.startswith('._') and not x.startswith('mir') and not x.startswith('med')]

    do_no_use = []

    max_feat = np.zeros(66)
    min_feat = np.ones(66)*1000

    max_voc = np.zeros(513)
    min_voc = np.ones(513)*1000

    max_mix = np.zeros(513)
    min_mix = np.ones(513)*1000    

    for voc_to_open in voc_list:

        voc_file = h5py.File(config.voice_dir+voc_to_open, "r")

        voc_stft = voc_file['voc_stft']

        feats = np.array(voc_file['feats'])

        f0 = feats[:,-2]

        med = np.median(f0[f0 > 0])

        f0[f0==0] = med

        feats[:,-2] = f0

        if np.isnan(feats).any():
            do_no_use.append(voc_to_open)

    #     maxi_voc_stft = np.array(voc_stft).max(axis=0)

    #     # if np.array(feats).min()<0:
    #     #     import pdb;pdb.set_trace()

    #     for i in range(len(maxi_voc_stft)):
    #         if maxi_voc_stft[i]>max_voc[i]:
    #             max_voc[i] = maxi_voc_stft[i]

    #     mini_voc_stft = np.array(voc_stft).min(axis=0)

    #     for i in range(len(mini_voc_stft)):
    #         if mini_voc_stft[i]<min_voc[i]:
    #             min_voc[i] = mini_voc_stft[i]

    #     maxi_voc_feat = np.array(feats).max(axis=0)

    #     for i in range(len(maxi_voc_feat)):
    #         if maxi_voc_feat[i]>max_feat[i]:
    #             max_feat[i] = maxi_voc_feat[i]

    #     mini_voc_feat = np.array(feats).min(axis=0)

    #     for i in range(len(mini_voc_feat)):
    #         if mini_voc_feat[i]<min_feat[i]:
    #             min_feat[i] = mini_voc_feat[i]   

    # for voc_to_open in back_list:

    #     voc_file = h5py.File(config.backing_dir+voc_to_open, "r")

    #     voc_stft = voc_file["back_stft"]

    #     maxi_voc_stft = np.array(voc_stft).max(axis=0)

    #     # if np.array(feats).min()<0:
    #     #     import pdb;pdb.set_trace()

    #     for i in range(len(maxi_voc_stft)):
    #         if maxi_voc_stft[i]>max_mix[i]:
    #             max_mix[i] = maxi_voc_stft[i]

    #     mini_voc_stft = np.array(voc_stft).min(axis=0)

    #     for i in range(len(mini_voc_stft)):
    #         if mini_voc_stft[i]<min_mix[i]:
    #             min_mix[i] = mini_voc_stft[i]

    # hdf5_file = h5py.File(config.stat_dir+'stats.hdf5', mode='w')

    # hdf5_file.create_dataset("feats_maximus", [66], np.float32) 
    # hdf5_file.create_dataset("feats_minimus", [66], np.float32)   
    # hdf5_file.create_dataset("voc_stft_maximus", [513], np.float32) 
    # hdf5_file.create_dataset("voc_stft_minimus", [513], np.float32)   
    # hdf5_file.create_dataset("back_stft_maximus", [513], np.float32) 
    # hdf5_file.create_dataset("back_stft_minimus", [513], np.float32)   

    # hdf5_file["feats_maximus"][:] = max_feat
    # hdf5_file["feats_minimus"][:] = min_feat
    # hdf5_file["voc_stft_maximus"][:] = max_voc
    # hdf5_file["voc_stft_minimus"][:] = min_voc
    # hdf5_file["back_stft_maximus"][:] = max_mix
    # hdf5_file["back_stft_minimus"][:] = min_mix

    import pdb;pdb.set_trace()

    # hdf5_file.close()


def get_stats_phonems():

    phon=collections.Counter([])

    voc_list = [x for x in os.listdir(config.voice_dir) if x.endswith('.hdf5') and x.startswith('nus') and not x.startswith('nus_KENN') and not x == 'nus_MCUR_read_17.hdf5']

    for voc_to_open in voc_list:

        voc_file = h5py.File(config.voice_dir+voc_to_open, "r")
        pho_target = np.array(voc_file["phonemes"])
        phon += collections.Counter(pho_target)
    phonemas_weights = np.zeros(41)
    for pho in phon:
        phonemas_weights[pho] = phon[pho]

    phonemas_above_threshold = [config.phonemas[x[0]] for x in np.argwhere(phonemas_weights>70000)]

    pho_order = phonemas_weights.argsort()

    # phonemas_weights = 1.0/phonemas_weights
    # phonemas_weights = phonemas_weights/sum(phonemas_weights)
    import pdb;pdb.set_trace()


def main():
    # gen_train_val()
    # get_stats()
    gen = data_gen_full('val', sec_mode = 0)
    while True :
        start_time = time.time()
        mix_in, singer_targs, voc_out, f0_out, pho_targs = next(gen)
        print(time.time()-start_time)

    #     plt.subplot(411)
    #     plt.imshow(np.log(1+inputs.reshape(-1,513).T),aspect='auto',origin='lower')
    #     plt.subplot(412)
    #     plt.imshow(targets.reshape(-1,66)[:,:64].T,aspect='auto',origin='lower')
    #     plt.subplot(413)
    #     plt.plot(targets.reshape(-1,66)[:,-2])
    #     plt.subplot(414)
    #     plt.plot(targets.reshape(-1,66)[:,-1])

    #     plt.show()
    #     # vg = val_generator()
    #     # gen = get_batches()


        import pdb;pdb.set_trace()


if __name__ == '__main__':
    main()