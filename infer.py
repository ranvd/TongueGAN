import imp


import cv2
import os
import argparse

import tensorflow as tf

from model import Pix2Pix
from dataset import create_dataset, augmentation, create_mask

# ------------------- arg -------------------
parser = argparse.ArgumentParser()

parser.add_argument('--model-name', type=str, default="pix2pix", help="name of model(use in saving weight respectively)")
parser.add_argument('--dataset', type=str, default="mask", help="dataset 的路徑")
parser.add_argument('--batch', type=int, default=8, help="batch size")
parser.add_argument('--image-size', type=int, default=512,
                    help="圖片要resize的大小(一個整數) eg. 512=512x512 ")

parser.add_argument('--resume-dir', type=str, default="./checkpoint", help="the direction of saving checkpoint")

args = parser.parse_args()

# =================== main ===================

dataset = create_mask(args)
dataset = dataset.map(lambda x: x/255)

model = Pix2Pix()
model.compile(G_opter=tf.keras.optimizers.Adam(2e-4, beta_1=0.5), 
              D_opter=tf.keras.optimizers.Adam(2e-4, beta_1=0.5))

# checkpoint_dir = os.path.join(args.resume_dir, args.model_name)
checkpoint = tf.train.Checkpoint(G_opter=model.G_opter,
                                 D_opter=model.D_opter,
                                 G_model=model.G,
                                 D_model=model.D)

checkpoint.restore(args.resume_dir)
print("checkpoint counter", checkpoint.save_counter)

counter = 0
for data in dataset:
    mask, image = data
    output = model.G(mask)
    for gen_img in output:
        cv2.imwrite("{}.png".format(counter),cv2.cvtColor(gen_img.numpy()*255, cv2.COLOR_RGB2BGR))
        counter += 1
    break