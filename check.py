import cv2
import os
import shutil
import argparse
from tqdm import tqdm
from hparams import hparams, get_image_list

parser = argparse.ArgumentParser(description='dddd')

parser.add_argument("--data_root", help="root", required=True, type=str)
parser.add_argument("--split", help="split", required=True, type=str)
args = parser.parse_args()
split = args.split
root = args.data_root

filelist = []
with open('filelists/{}.txt'.format(split)) as f:
    for line in f:
        line = line.strip()
        if ' ' in line: line = line.split()[0]
        filelist.append(line)

print(f'filelist: {len(filelist)} ')

f_list = []
for p in tqdm(filelist):
    filepath = os.path.join(root, p, "0.jpg")
    image = cv2.imread(filepath)
    if image.shape[1] < 288:
        d_path = os.path.join(root, p)
        shutil.rmtree(d_path)
        line = p+'\n'
        f_list.append(line)

print(f'delete file: {len(f_list)}')

with open(f'./filelists/del_{split}', 'w') as f:
    f.writelines(f_list)
