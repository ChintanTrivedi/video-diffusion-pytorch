import glob
from PIL import Image
import os
import json
from tqdm import tqdm
import numpy as np


def flatten(features):
    flattened_features = []
    for attribute in features.items():
        value = attribute[1]
        flattened_features.extend(value) if isinstance(value, list) else flattened_features.append(value)
    return np.array(flattened_features)


carla_drivegan_dataset = 'D:/PycharmProjects/video-diffusion-pytorch-lucidrains/datasets/drivegan_holdout'
output_directory = 'D:/PycharmProjects/video-diffusion-pytorch-lucidrains/datasets/drivegan_conditional_video'
os.makedirs(output_directory) if not os.path.exists(output_directory) else None

for folder in tqdm(os.listdir(carla_drivegan_dataset)):
    episodes = os.listdir(os.path.join(carla_drivegan_dataset, folder))
    for episode in tqdm(episodes):
        try:
            gif_images_path = os.path.join(carla_drivegan_dataset, folder, episode)

            fp_in = os.path.join(gif_images_path, "*.png")
            imgs = [Image.open(f) for f in
                    sorted(glob.glob(fp_in), key=lambda x: int(os.path.basename(x.replace('.png', '')).split(' ')[0]))]

            action_info = json.load(open(os.path.join(gif_images_path, 'info.json')))['data']
            for sub_idx in range(8):
                sub_imgs = iter(imgs[sub_idx * 10: sub_idx * 10 + 10])
                fp_out = os.path.join(output_directory, f'{folder}_{episode}_{sub_idx}')
                os.makedirs(fp_out) if not os.path.exists(fp_out) else None

                # extract first image from iterator
                img = next(sub_imgs)

                # conditioned on the start image
                fp_out_image = os.path.join(fp_out, 'start.png')
                fp_out_gif = os.path.join(fp_out, 'forward.gif')
                fp_out_action = os.path.join(fp_out, 'action.npy')

                # conditioned on action taken at initial frame
                action = flatten(action_info[sub_idx * 10])

                # save everything to disk
                img.save(fp_out_image)
                img.save(fp=fp_out_gif, format='GIF', append_images=sub_imgs, save_all=True, duration=200, loop=0)
                np.save(fp_out_action, action)
        except:
            continue
