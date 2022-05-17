from video_diffusion_carla import Unet3D, GaussianDiffusion, Trainer

model = Unet3D(
    dim=8,
    cond_dim=512 + 6,
    dim_mults=(1, 2, 4, 8),
)

diffusion = GaussianDiffusion(
    model,
    image_size=8,
    num_frames=10,
    timesteps=10,  # number of steps
    loss_type='l1'  # L1 or L2
).cuda()

trainer = Trainer(
    diffusion,
    'D:/PycharmProjects/video-diffusion-pytorch-lucidrains/datasets/drivegan_conditional_video',
    train_batch_size=1,
    train_lr=1e-4,
    num_sample_rows=1,
    cond_sampling_folder='D:/PycharmProjects/video-diffusion-pytorch-lucidrains/datasets/test_sampling',
    save_and_sample_every=1,
    train_num_steps=700000,  # total training steps
    gradient_accumulate_every=2,  # gradient accumulation steps
    ema_decay=0.995,  # exponential moving average decay
    amp=True  # turn on mixed precision
)

trainer.train()
