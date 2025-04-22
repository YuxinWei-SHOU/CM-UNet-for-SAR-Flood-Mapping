import numpy as np
from pathlib import Path
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from osgeo import gdal
import wandb
import logging

from Utils.path_hyperparameter import ph

def save_model(model, path, epoch, mode, optimizer=None):
    """
    Save the best model when the best metric appears during evaluation,
    or save checkpoints at every specified interval during evaluation.

    Parameters:
        model(class): The neural network we built
        path(str): Path where the model will be saved
        epoch(int): The training epoch when the model is saved
        mode(str): Ensure whether to save the best model or checkpoint, should be checkpoint, loss or f1score
        optimizer(class, optional): The optimizer used during training, needed when saving checkpoints

    Returns:
        No return value
    """
    # Create directory to save the model
    Path(path).mkdir(parents=True, exist_ok=True)
    # Get the current local time and convert it to a readable string format
    localtime = time.asctime(time.localtime(time.time()))
    if mode == 'checkpoint':  # If it's checkpoint mode
        state_dict = {'net': model.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(state_dict, str(path + f'checkpoint_epoch{epoch}_{localtime}.pth'))
    else:  # Save the best model
        torch.save(model.state_dict(), str(path + f'best_{mode}_epoch{epoch}_{localtime}.pth'))
    logging.info(f'best {mode} model {epoch} saved at {localtime}!')

def train_val_test(
        mode, dataset_name,
        dataloader, device, log_wandb, net, optimizer, total_step,
        lr, criterion, metric_collection, to_pilimg, epoch,
        warmup_lr=None, grad_scaler=None,
        best_metrics=None, checkpoint_path=None,
        best_loss_model_path=None, best_f1_model_path=None, non_improved_epoch=None
):
    """
    Train or evaluate the model on the specified dataset,
    Note that parameters [warmup_lr, grad_scaler] are required for training,
    parameters [best_metrics, checkpoint_path, best_loss_model_path, non_improved_epoch] are required for evaluation.

    Parameters:
        mode(str): Ensure whether it is for training or evaluation, should be 'train' or 'val'
        dataset_name(str): Name of the dataset
        dataloader(class): Dataloader corresponding to the mode and specified dataset
        device(str): The device where the model runs
        log_wandb(class): The class used for logging hyperparameters, metrics, and outputs
        net(class): The neural network we built
        optimizer(class): The optimizer used for training
        total_step(int): The total training steps
        lr(float): Learning rate
        criterion(class): Loss function
        metric_collection(class): Metric calculator
        to_pilimg(function): Function to convert arrays to images
        epoch(int): The training epoch
        warmup_lr(list, optional): The learning rate for each step during the warmup phase
        grad_scaler(class, optional): Gradient scaling when using mixed precision
        best_metrics(list, optional): Best metrics during evaluation
        checkpoint_path(str, optional): Checkpoint save path
        best_loss_model_path(str, optional): Best loss model save path
        non_improved_epoch(int, optional): The number of epochs with no improvement in the best metric

    Returns:
        Returns different modified input parameters depending on the mode,
        when mode = 'train', returns log_wandb, net, optimizer, grad_scaler, total_step, lr
        when mode = 'val', returns log_wandb, net, optimizer, total_step, lr, best_metrics, non_improved_epoch
    """
    # Ensure mode is 'train' or 'val'
    assert mode in ['train', 'val'], 'mode should be train, val'
    epoch_loss = 0
    if mode == 'train':
        net.train()  # Set model to training mode
    else:
        net.eval()  # Set model to evaluation mode
    logging.info(f'SET model mode to {mode}!')
    batch_iter = 0

    # Use tqdm to display progress bar
    tbar = tqdm(dataloader)
    n_iter = len(dataloader)
    sample_batch = np.random.randint(low=0, high=n_iter)

    # Define sample_name to log sample image names
    sample_name = None

    for i, (image, labels, name) in enumerate(tbar):
        tbar.set_description(
            "epoch {} info ".format(epoch) + str(batch_iter) + " - " + str(batch_iter + ph.batch_size))
        batch_iter = batch_iter + ph.batch_size
        total_step += 1

        if mode == 'train':
            optimizer.zero_grad()
            # if total_step < ph.warm_up_step:     Uncomment if warmup is needed
            #     for g in optimizer.param_groups:
            #         g['lr'] = warmup_lr[total_step]

        image = image.float().to(device)
        labels = labels.float().to(device)

        b, c, h, w = image.shape
        # Downsample image and label
        image = F.interpolate(image, size=(h // ph.downsample_raito, w // ph.downsample_raito), mode='bilinear', align_corners=False)
        labels = F.interpolate(labels.unsqueeze(1), size=(h // ph.downsample_raito, w // ph.downsample_raito), mode='bilinear', align_corners=False).squeeze(1)

        if mode == 'train':
            with torch.cuda.amp.autocast():
                preds = net(image)
                #loss_change, diceloss, bceloss = criterion(preds, labels)
                bceloss = criterion(preds, labels)
            cd_loss = bceloss.mean()
            #cd_loss = bceloss.mean()
            grad_scaler.scale(cd_loss).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1, norm_type=2)
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            preds = net(image)
            #loss_change, diceloss, bceloss = criterion(preds, labels)
            bceloss = criterion(preds, labels)
            cd_loss = bceloss.mean()  # Loss for the current batch, for 2 images, overall evaluation metrics do not look at this

        epoch_loss += cd_loss

        preds = torch.sigmoid(preds)

        if i == sample_batch:
            sample_index = np.random.randint(low=0, high=image.shape[0])
            t1_img_log = torch.round(image[sample_index]).cpu().clone().float()
            label_log = torch.round(labels[sample_index]).cpu().clone().float()
            pred_log = torch.round(preds[sample_index]).cpu().clone().float()
            sample_name = name[sample_index]  # Record the current sample image name

        batch_metrics = metric_collection(preds.float(), labels.int().unsqueeze(1))

        # Log batch metrics
        log_wandb.log({
            f'{mode} loss': cd_loss,
            f'{mode} accuracy': batch_metrics['accuracy'],
            f'{mode} precision': batch_metrics['precision'],
            f'{mode} recall': batch_metrics['recall'],
            f'{mode} f1score': batch_metrics['f1score'],
            f'{mode} miou': batch_metrics['miou'],
            'learning rate': optimizer.param_groups[0]['lr'],
            #f'{mode} loss_dice': diceloss,
            f'{mode} loss_bce': bceloss,
            'step': total_step,
            'epoch': epoch
        })

        del image, labels

    epoch_metrics = metric_collection.compute()
    epoch_loss /= n_iter

    # Print and log metrics for each epoch (these are the metrics we need to focus on)
    print(f"{mode} miou: {epoch_metrics['miou']}")
    print(f"{mode} accuracy: {epoch_metrics['accuracy']}")
    print(f"{mode} precision: {epoch_metrics['precision']}")
    print(f"{mode} recall: {epoch_metrics['recall']}")
    print(f"{mode} f1score: {epoch_metrics['f1score']}")
    print(f'{mode} epoch loss: {epoch_loss}')

    for k in epoch_metrics.keys():
        log_wandb.log({f'epoch_{mode}_{str(k)}': epoch_metrics[k], 'epoch': epoch})
    metric_collection.reset()
    log_wandb.log({f'epoch_{mode}_loss': epoch_loss, 'epoch': epoch})

    # Log image names here
    log_wandb.log({
        f'{mode} t1_images {sample_name}': wandb.Image(t1_img_log),  # Log sample image names
        f'{mode} masks': {
            f'label {sample_name}': wandb.Image(to_pilimg(label_log)),
            f'pred {sample_name}': wandb.Image(to_pilimg(pred_log)),
        },
        'epoch': epoch
    })

    if mode == 'val':  # On validation set
        if epoch_loss < best_metrics['lowest_loss']:  # If current epoch loss is lower than the previous best loss, update best loss
            best_metrics['lowest_loss'] = epoch_loss  # Update lowest loss in best_metrics
            best_metrics['best_epoch_loss'] = epoch  # Record the best epoch
            if ph.save_best_model:  # If set to save the best model, call save_model to save the current model
                save_model(net, best_loss_model_path, epoch, 'loss')

        # Save the best model based on F1 score
        if epoch_metrics['f1score'] > best_metrics['highest_f1']:  # If current epoch F1 score is higher than previous best, update
            best_metrics['highest_f1'] = epoch_metrics['f1score']  # Update highest F1 score in best_metrics
            best_metrics['best_epoch_f1'] = epoch  # Record the best F1 epoch
            if ph.save_best_model:  # If set to save the best model, call save_model to save the current model
                save_model(net, best_f1_model_path, epoch, 'f1score')

        # else:  # If current epoch performance has not improved, increase non_improved_epoch counter   Uncomment if warmup is needed
        #     non_improved_epoch += 1
        #     if non_improved_epoch == ph.patience:  # Similar to early stopping
        #         lr *= ph.factor  # Reduce the learning rate by a factor of ph.factor
        #         for g in optimizer.param_groups:
        #             g['lr'] = lr
        #         non_improved_epoch = 0

        if (epoch + 1) % ph.save_interval == 0 and ph.save_checkpoint:
            save_model(net, checkpoint_path, epoch, 'checkpoint', optimizer=optimizer)

    if mode == 'train':
        return log_wandb, net, optimizer, grad_scaler, total_step, lr
    elif mode == 'val':
        return log_wandb, net, optimizer, total_step, lr, best_metrics, non_improved_epoch
