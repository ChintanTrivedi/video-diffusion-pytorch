from video_diffusion_carla import Unet3D, GaussianDiffusion, Trainer

model = Unet3D(
    dim=128,
    cond_dim=512 + 32,
    dim_mults=(1, 2, 4, 8),
)

diffusion = GaussianDiffusion(
    model,
    image_size=128,
    num_frames=10,
    timesteps=1000,  # number of steps
    loss_type='l1'  # L1 or L2
).cuda()

trainer = Trainer(
    diffusion,
    './data',
    train_batch_size=4,
    train_lr=2e-5,
    save_and_sample_every=5000,
    train_num_steps=700000,  # total training steps
    gradient_accumulate_every=2,  # gradient accumulation steps
    ema_decay=0.995,  # exponential moving average decay
    amp=True  # turn on mixed precision
)

trainer.train()
