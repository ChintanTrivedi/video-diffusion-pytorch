import glob
from PIL import Image
import os
from tqdm import tqdm

carla_drivegan_dataset = './datasets/carla_drivegan'
dataset_images = '64053'

for folder in tqdm(os.listdir(os.path.join(carla_drivegan_dataset, dataset_images))):
    folder_path = os.path.join(os.path.join(carla_drivegan_dataset, dataset_images), folder)

    fp_in = os.path.join(folder_path, "*.png")
    fp_out = os.path.join(carla_drivegan_dataset, folder + '.gif')

    imgs = (Image.open(f) for f in sorted(glob.glob(fp_in)))
    img = next(imgs)  # extract first image from iterator
    img.save(fp=fp_out, format='GIF', append_images=imgs, save_all=True, duration=200, loop=0)

dim = 128
init_dim = dim // 3 * 2
dim_mults = (1, 2, 4, 8)
dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
in_out = list(zip(dims[:-1], dims[1:]))
