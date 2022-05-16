import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision import transforms as T

from video_diffusion_pytorch import Unet3D, GaussianDiffusion


def unnormalize_img(t):
    return (t + 1) * 0.5


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def video_tensor_to_gif(tensor, path, duration=120, loop=0, optimize=True):
    images = map(T.ToPILImage(), tensor.unbind(dim=1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all=True, append_images=rest_imgs, duration=duration, loop=loop, optimize=optimize)
    return images


model = Unet3D(dim=128, dim_mults=(1, 2, 4, 8), )
model = nn.DataParallel(model)

state = torch.load(str('D:/PycharmProjects/video-diffusion-pytorch/results/model.pt'))['model']
for k in list(state.keys()):
    if "denoise_fn." in k:
        state[k.replace("denoise_fn.", "")] = state[k]
    del state[k]

model.load_state_dict(state)

diffusion = GaussianDiffusion(
    model,
    image_size=128,
    num_frames=10,
    timesteps=1000,  # number of steps
    loss_type='l1'  # L1 or L2
).cuda()

num_sample_rows = 4

num_samples = num_sample_rows ** 2
batches = num_to_groups(num_samples, batch_size=2)

all_videos_list = list(map(lambda n: diffusion.sample(batch_size=n), batches))
all_videos_list = torch.cat(all_videos_list, dim=0)
all_videos_list = unnormalize_img(all_videos_list)

all_videos_list = F.pad(all_videos_list, (2, 2, 2, 2))

one_gif = rearrange(all_videos_list, '(i j) c f h w -> c f (i h) (j w)', i=num_sample_rows)
video_path = 'results/carla_samples_50k_steps.gif'
video_tensor_to_gif(one_gif, video_path)
