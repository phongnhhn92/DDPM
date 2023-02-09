import torch, os
import numpy as np
from torch.utils.data import DataLoader
from opt import config_parser

# model
from model import Unet, GaussianDiffusion

# dataset
from torchvision import datasets
import torchvision.transforms.transforms as T
from torchvision.utils import save_image
# optimizer, scheduler, visualization
from utils import *
from torch.optim.lr_scheduler import CosineAnnealingLR

# pytorch-lightning
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning import LightningModule, Trainer, loggers


class Project_name(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = Unet(
                dim = 64,
                dim_mults = (1, 2, 4, 8)
        )

        self.diffusion = GaussianDiffusion(
                self.model,
                image_size = 32,
                timesteps = self.hparams.num_step,   # number of steps
                sampling_timesteps = self.hparams.num_step,
                loss_type = 'l1'    # L1 or L2
        )
    # def decode_batch():
    #     return None
    
    def forward(self):
        return None

    def prepare_data(self):
        transforms = T.Compose([T.RandomHorizontalFlip(), T.ToTensor()])
        self.train_dataset = datasets.CIFAR100("./data/CIFAR", download=True, train=True, transform=transforms)
        # self.test_dataset = datasets.CIFAR100("./data/CIFAR", download=True, train=False, transform=transforms)
        # indices = list(range(len(self.train_dataset)))
        # np.random.shuffle(indices)

        # split = int(np.floor(0.2* len(self.train_dataset)))
        # self.train_sample = SubsetRandomSampler(indices[:split])
        # self.test_sample = SubsetRandomSampler(indices[split:])
    
    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, [self.diffusion])
        scheduler = get_scheduler(self.hparams, self.optimizer)

        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=4, batch_size=self.hparams.batchSize, shuffle=True)

    # def val_dataloader(self):
    #     return DataLoader(self.train_dataset, shuffle=False, num_workers=1, batch_size=1)
        
    # def test_dataloader(self):
    #     return DataLoader(self.test_dataloader, num_workers=1, batch_size=1, sampler=self.test_sample)

    def training_step(self, batch, batch_nb):
        learning_rate = self.optimizer.param_groups[0]['lr']
        self.log('lr', learning_rate, prog_bar=True, logger=True)
        imgs = batch[0]
        B,C,H,W = imgs.shape

        if C != 3:
            imgs = torch.repeat_interleave(imgs, repeats=3, dim=1)

        loss = self.diffusion(imgs)
        self.log('train_loss', loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)

        if batch_nb % self.hparams.num_vis_step == 0:
            sampled_images = self.diffusion.sample(batch_size = self.hparams.num_vis)
            saving_folder = os.path.join(self.hparams.sample_dir, f'{args.expname}',f'{self.logger.version}')
            if os.path.exists(saving_folder) is not True:
                os.makedirs(os.path.join(saving_folder,'generated'))

            save_image(sampled_images, os.path.join(saving_folder,'generated', f'{self.current_epoch:04}_{batch_nb:04}.png'))

            if self.hparams.save_imgs:
                if os.path.exists(os.path.join(saving_folder,'GT')) is not True:
                    os.makedirs(os.path.join(saving_folder,'GT'))
                save_image(imgs, os.path.join(saving_folder,'GT', f'{self.current_epoch:04}_{batch_nb:04}.png'))

        return loss

    # def validation_step(self, batch, batch_nb):
    #     log = {}
    #     return log
        

    # def validation_epoch_end(self, outputs):
    #     return None
    
    def save_ckpt(self):
        print('saved!')
        
    

if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    args = config_parser()
    system = Project_name(args)

    checkpoint_callback = ModelCheckpoint(dirpath= f'{args.log_dir}/{args.expname}/ckpts/',
                                          filename= '{epoch:02d}',
                                          monitor='train_loss',
                                          mode='max',
                                          save_top_k=3)
    bar = TQDMProgressBar(refresh_rate=1000 if args.num_gpus > 1 else 1)
    logger = loggers.TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.expname,
    )

    trainer = Trainer(max_epochs=args.num_epochs,
                      callbacks=[checkpoint_callback,bar],
                      logger=logger,
                      enable_model_summary=False,
                      accelerator="gpu" if args.num_gpus >= 1 else "cpu",
                      devices=args.num_gpus if args.num_gpus is not None else 0,   
                      gradient_clip_val=args.grad_clip,                                   
                      #num_sanity_val_steps=1,
                      #check_val_every_n_epoch = max(args.num_epochs//args.N_vis,1),
                      benchmark=True,)

    trainer.fit(system)
    system.save_ckpt()
    torch.cuda.empty_cache()