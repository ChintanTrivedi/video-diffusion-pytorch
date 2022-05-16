import glob
from PIL import Image
import os
from tqdm import tqdm

carla_drivegan_dataset = "/media/storage/Chintan/video-diffusion-pytorch/datasets/drivegan"
dataset_subfolders = ['64054', '64055', '64056', '64057', '64058']

for dataset_subfolder in dataset_subfolders:
    for gif_index in tqdm(os.listdir(os.path.join(carla_drivegan_dataset, dataset_subfolder))):
        try:
            folder_path = os.path.join(carla_drivegan_dataset, dataset_subfolder, gif_index)

            fp_in = os.path.join(folder_path, "*.png")
            fp_out = os.path.join(carla_drivegan_dataset + '_gif', str(dataset_subfolder) + str(gif_index) + '.gif')

            imgs = (Image.open(f) for f in
                    sorted(glob.glob(fp_in), key=lambda x: int(os.path.basename(x.replace('.png', '')).split(' ')[0])))
            img = next(imgs)  # extract first image from iterator
            img.save(fp=fp_out, format='GIF', append_images=imgs, save_all=True, duration=200, loop=0)
        except:
            continue
