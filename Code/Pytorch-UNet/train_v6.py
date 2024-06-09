import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.nn import DataParallel

import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset, CustomDataset
from utils.dice_score import dice_loss


import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm

# Define la clase EarlyStopping

class EarlyStopping:
    def __init__(self, patience=5, delta=0, checkpoint_path='checkpoint.pth'):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.checkpoint_path = checkpoint_path

    def __call__(self, val_score, model):
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(val_score, model)
        elif val_score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.save_checkpoint(val_score, model)
            self.counter = 0

    def save_checkpoint(self, val_score, model):
        if self.checkpoint_path:
            torch.save(model.state_dict(), self.checkpoint_path)

@torch.inference_mode()
def evaluate22(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    mse_score = 0
   
    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
       
        #print("Number of validation batches:", num_val_batches)
        num_val_batches = len(dataloader)
        #print("Number of validation batches:", num_val_batches)
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, true_values = batch['data'], batch['target']
           
            #print("Batch dimensions:", batch['data'].shape, batch['target'].shape)

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            true_values = true_values.to(device=device, dtype=torch.float32)
            
            # predict the values
            predicted_values = net(image)
            nonzero_indices = torch.where(true_values.view(-1) != 0)[0]

	        # Seleccionar solo los valores donde true_values es distinto de 0
            predicted_nonzero = predicted_values.view(-1)[nonzero_indices]
            true_nonzero = true_values.view(-1)[nonzero_indices]

            # Imprimir los valores predichos y verdaderos
            #print("Predicted values:", predicted_nonzero.detach().cpu().numpy())
            #print("True values:", true_nonzero.detach().cpu().numpy())

            # Calcular la diferencia total entre los valores reales y predichos
            total_difference = torch.sum(torch.abs(predicted_nonzero - true_nonzero))

            #print("Diferencia total:", total_difference.item())
            
            
            
            



            #print("Predicted values:", predicted_values.view(-1)[:5].detach().cpu().numpy())
            #print("True values:", true_values.view(-1)[:5].detach().cpu().numpy())

            # compute the MSE score only for non-zero elements
            #non_zero_mask = true_values != 0

            # Squeeze para eliminar la dimensión adicional
            #predicted_values_squeezed = predicted_values.squeeze(1)

            #mse_score += F.mse_loss(predicted_values_squeezed[non_zero_mask], true_values[non_zero_mask], reduction='mean')
            mse_score += F.mse_loss(predicted_nonzero, true_nonzero)


    net.train()
    print('*********** ',mse_score / max(num_val_batches, 1))
    return mse_score / max(num_val_batches, 1)




def get_available_devices():
    devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())] 
    return devices if devices else ['cpu']







dir_img = Path('/opt/share/MERIDA/DATASET-TESTALL2/IN/')
dir_mask = Path('/opt/share/MERIDA/DATASET-TESTALL2/OUT/')
dir_checkpoint = Path('/opt/share/MERIDA/Code/Pytorch-UNet/checkpoints2/')



def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 2,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    # 1. Create dataset
    try:
        dataset = CustomDataset(dir_img, dir_mask)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    #criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    #criterion = nn.MSELoss()
    criterion = nn.SmoothL1Loss()
    global_step = 0
    
    # Crea una instancia de EarlyStopping
    
    early_stopping = EarlyStopping(patience=50, checkpoint_path='/opt/share/MERIDA/Code/Pytorch-UNet/checkpoints/early_stop_checkpoint.pth')



    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['data'], batch['target']
                #images = images.permute(0, 3, 1, 2)
                #print(images.shape,'ja',true_masks.shape)

                

                assert images.shape[1] == model.module.n_channels, \
                    f'Network has been defined with {model.module.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                  masks_pred = model(images)

                  # Calcular la pérdida MSE
                  #loss = criterion(masks_pred.squeeze(1), true_masks.float())
                  #loss = criterion(masks_pred, true_masks.float())

                  # Calcular la pérdida MSE solo para los elementos no cero
                  non_zero_mask = true_masks != 0
                  predicted_non_zero = masks_pred[:, 0, :, :][non_zero_mask]
                  true_non_zero = true_masks[non_zero_mask].float()
                  loss = criterion(predicted_non_zero, true_non_zero)

                 

                  

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                #print('EL DIVISION STEP ES',division_step,global_step % division_step)

                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())
                        
                        val_score = evaluate22(model, val_loader, device, amp)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            #state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')
            
        val_score = evaluate22(model, val_loader, device, amp)
        scheduler.step(val_score)
        early_stopping(val_score, model)

        if early_stopping.early_stop:
            logging.info("Early stopping")
            break


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #devices =[torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())] 
    #logging.info(f'Using device {device}')
    devices = get_available_devices()
    logging.info(f'Available devices: {devices}')
    
    

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=8, n_classes=1, bilinear=args.bilinear).to(device)
    #model = model.to(memory_format=torch.channels_last)
    
    if len(devices) > 1:
        logging.info(f'Using dataParallel with devices: {devices}')
        model = torch.nn.DataParallel(model, device_ids=devices)
    


    #logging.info(f'Network:\n'
    #             f'\t{model.n_channels} input channels\n'
    #             f'\t{model.n_classes} output channels (classes)\n'
    #             f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    #model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size // len(devices),
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )


