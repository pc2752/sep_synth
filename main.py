import models
import tensorflow as tf
import argparse
import os, sys
import config
import utils
import numpy as np
import mir_eval

def train(_):
    model = models.MultiSynth()
    model.train()

def train_phone(_):
    model = models.Phone_Net()
    model.train()
def eval_hdf5_file(file_name, file_name_singer):
    model = models.MultiSynth()
    model.test_file_hdf5(file_name, file_name_singer)
def eval_wav_file(file_name, file_name_singer):
    model = models.MultiSynth()
    model.test_file_wav(file_name, file_name_singer)


if __name__ == '__main__':
    if len(sys.argv)<2 or sys.argv[1] == '-help' or sys.argv[1] == '--help' or sys.argv[1] == '--h' or sys.argv[1] == '-h':
        print("%s --help or -h or --h or -help to see this menu" % sys.argv[0])
        print("%s --train or -t or --t or -train to train the model" % sys.argv[0])
        print("%s -e or --e or -eval or --eval  <filename> to evaluate an hdf5 file" % sys.argv[0])
        print("%s -v or --v or -val or --val <filename> to calculate metrics for entire dataset and save to given filename" % sys.argv[0])
        print("%s -w or --w or -wavfile or --wavfile <filename> <save_path> to evaluate wavefile and save CSV" % sys.argv[0])
        print("%s -wf or --wf or -wavfolder or --wavolder <foldername> <save_path> to evaluate all wavefiles in the folder and save CSV" % sys.argv[0])
    else:
        if sys.argv[1] == '-train' or sys.argv[1] == '--train' or sys.argv[1] == '--t' or sys.argv[1] == '-t':
            print("Training")
            tf.app.run(main=train)

        elif sys.argv[1] == '-e' or sys.argv[1] == '--e' or sys.argv[1] == '--eval' or sys.argv[1] == '-eval':
            if len(sys.argv)<3:
                print("Please give a file to evaluate")
                print([x for x in os.listdir(config.voice_dir) if x.startswith('nus') or x.startswith('casas') or x.startswith('med')])
            else:
                file_name = sys.argv[2]
                if not file_name.endswith('.hdf5'):
                    file_name = file_name+'.hdf5'
                if not file_name in [x for x in os.listdir(config.voice_dir) if x.startswith('nus') or x.startswith('casas') or x.startswith('med')]:
                    print("Currently only supporting hdf5 files which are in the dataset, will be expanded later.")
                    print([x for x in os.listdir(config.voice_dir) if x.startswith('nus' )or x.startswith('casas') or x.startswith('med')])
                else:
                    if len(sys.argv)<4:
                        print("Synthesizing with same singer.")
                        eval_wav_file(file_name, file_name)
                    else:
                        file_name_singer = sys.argv[3]
                        if not file_name_singer in [x for x in os.listdir(config.voice_dir) if x.startswith('nus') or x.startswith('casas') or x.startswith('med')]:
                            print("Currently only supporting hdf5 files which are in the dataset, will be expanded later.")
                            print([x for x in os.listdir(config.voice_dir) if x.startswith('nus' )or x.startswith('casas') or x.startswith('med')])
                        else:
                            print("Synthesizing second singer.")
                            eval_wav_file(file_name, file_name_singer)
        elif sys.argv[1] == '-w' or sys.argv[1] == '--w' or sys.argv[1] == '--wav' or sys.argv[1] == '-wav':
            if len(sys.argv)<3:
                print("Please give a wave file to evaluate")
            else:
                file_name = sys.argv[2]
                if len(sys.argv)<4:
                    print("Synthesizing with same singer.")
                    eval_wav_file(file_name, file_name)
                else:
                    file_name_singer = sys.argv[3]
                    if not file_name_singer in [x for x in os.listdir(config.voice_dir) if x.startswith('nus') or x.startswith('casas') or x.startswith('med')]:
                        print("Currently only supporting hdf5 files which are in the dataset, will be expanded later.")
                        print([x for x in os.listdir(config.voice_dir) if x.startswith('nus' )or x.startswith('casas') or x.startswith('med')])
                    else:
                        print("Synthesizing second singer.")
                        eval_wav_file(file_name, file_name_singer)
        elif sys.argv[1] == '-p' or sys.argv[1] == '--t' or sys.argv[1] == '--phone' or sys.argv[1] == '-phone':
            if len(sys.argv)<3:
                print("Training Phone")
                tf.app.run(main=train_phone)