import glob
import json
import os

import clip
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# pretrained CLP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def flatten(features):
    flattened_features = []
    for attribute in features.items():
        value = attribute[1]
        flattened_features.extend(value) if isinstance(value, list) else flattened_features.append(value)
    return np.array(flattened_features)


carla_drivegan_dataset = 'D:/PycharmProjects/video-diffusion-pytorch-lucidrains/datasets/drivegan_holdout'
dataset_subfolders = ['64053', '64054', '64055', '64056', '64057', '64058']

output_directory = 'D:/PycharmProjects/video-diffusion-pytorch-lucidrains/datasets/drivegan_conditional_video'
os.makedirs(output_directory) if not os.path.exists(output_directory) else None

for folder in tqdm(dataset_subfolders):
    episodes = os.listdir(os.path.join(carla_drivegan_dataset, folder))
    for episode in tqdm(episodes):
        try:
            gif_images_path = os.path.join(carla_drivegan_dataset, folder, episode)

            fp_in = os.path.join(gif_images_path, "*.png")
            imgs = [Image.open(f).resize((128, 128)) for f in
                    sorted(glob.glob(fp_in), key=lambda x: int(os.path.basename(x.replace('.png', '')).split(' ')[0]))]

            action_info = json.load(open(os.path.join(gif_images_path, 'info.json')))['data']
            for sub_idx in range(8):
                sub_imgs = iter(imgs[sub_idx * 10: sub_idx * 10 + 10])
                fp_out = os.path.join(output_directory, f'{folder}_{episode}_{sub_idx}')
                os.makedirs(fp_out) if not os.path.exists(fp_out) else None

                # extract first image from iterator
                img = next(sub_imgs)
                img_tensor = preprocess(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    img_emb = model.encode_image(img_tensor)
                    img_emb = img_emb.cpu().numpy().T

                # conditioned on the start image
                fp_out_image = os.path.join(fp_out, 'start.png')
                fp_out_image_emb = os.path.join(fp_out, 'start_emb.txt')
                fp_out_gif = os.path.join(fp_out, 'forward.gif')
                fp_out_action = os.path.join(fp_out, 'action.txt')

                # conditioned on action taken at initial frame
                action = flatten(action_info[sub_idx * 10])

                # save everything to disk
                img.save(fp_out_image)
                img.save(fp=fp_out_gif, format='GIF', append_images=sub_imgs, save_all=True, duration=200, loop=0)
                np.savetxt(fp_out_action, action)
                np.savetxt(fp_out_image_emb, img_emb)
        except:
            continue
