import torch, os
from torch.utils.data import DataLoader
from opt import config_parser

# optimizer, scheduler, visualization
from torch.optim.lr_scheduler import CosineAnnealingLR

# pytorch-lightning
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning import LightningModule, Trainer, loggers


class Project_name(LightningModule):
    def __init__(self, args):
        super(Project_name,self).__init__()
    
    # def decode_batch():
    #     return None
    
    def forward(self):
        return 

    def prepare_data(self):
        return 
    
    def configure_optimizers(self):
        return super().configure_optimizers()

    def train_dataloader(self):
        return super().train_dataloader()

    def val_dataloader(self):
        return super().val_dataloader()

    def training_step(self, *args, **kwargs):
        return super().training_step(*args, **kwargs)

    def validation_step(self, *args, **kwargs):
        return super().validation_step(*args, **kwargs)

    def validation_epoch_end(self, outputs):
        return super().validation_epoch_end(outputs)
    
    def save_ckpt(self):
        print('saved!')
        
    

if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    args = config_parser()
    system = Project_name(args)

    checkpoint_callback = ModelCheckpoint(dirpath= f'{args.log_dir}/{args.expname}/ckpts/',
                                          filename= '{epoch:02d}',
                                          monitor='val/PSNR',
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
                      num_sanity_val_steps=1,
                      check_val_every_n_epoch = max(args.num_epochs//args.N_vis,1),
                      benchmark=True,)

    trainer.fit(system)
    system.save_ckpt()
    torch.cuda.empty_cache()