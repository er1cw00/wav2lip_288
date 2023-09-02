
import os
import audio
import argparse
import traceback
import numpy as np
from pathlib import Path
from os import path
from tqdm import tqdm
from glob import glob
from hparams import hparams

parser = argparse.ArgumentParser(description='Code to train the expert lip-sync discriminator')

parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", required=True)

args = parser.parse_args()

def process_audio_file(vfile, args):
    p = Path(vfile)
    specpath = p.parent.joinpath('spec.npy')
    wav = audio.load_wav(vfile, hparams.sample_rate)
    mel = audio.melspectrogram(wav).T
    np.save(specpath, mel)
    

filelist = glob(path.join(args.data_root, '*/*/*.wav'))
print(f'filelist: {len(filelist)}')
for vfile in tqdm(filelist):
    try:
        process_audio_file(vfile, args)
    except KeyboardInterrupt:
        exit(0)
    except:
        traceback.print_exc()
        continue
    
