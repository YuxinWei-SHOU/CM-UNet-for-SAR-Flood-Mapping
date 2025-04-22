import sys  # System-specific parameters and functions
import time  # Time-related functions
import ipdb  # Interactive Python debugger
import numpy as np  # Numerical computation library
from torch import optim  # Optimizer module from PyTorch
import torchvision.transforms as T  # Computer vision data processing library
from torch.utils.data import DataLoader  # Data loading and batching tool
from Utils.data_loading import BasicDataset  # Dataset loading utility
from Utils.path_hyperparameter import ph  # Path and hyperparameter configuration
import torch  # PyTorch main library
from Utils.losses import FCCDN_loss_without_seg  # Loss functions
import os  # Operating system related functions
import logging  # Logging module
import random  # Random number generation module
import wandb  # Weights and Biases, experiment tracking and visualization tool
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, JaccardIndex  # Evaluation metrics tools
from Utils.utils import train_val_test  # Training and validation functions
#from Nets.resunet import ResUNet
#from Nets.TransUnet import TransUNet
#from Nets.BaseUnet import UNet
from Nets.CM_UNet import MambaUNet


# Using a random seed ensures that the sequence of random numbers generated is the same every time the code runs, thus yielding the same results
def random_seed(SEED):
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# Set the random seed and start the training process, with appropriate handling in case the user interrupts
def auto_experiment():
    random_seed(SEED=ph.random_seed)
    try:
        train_net(dataset_name=ph.dataset_name)
    except KeyboardInterrupt:
        logging.info('Interrupt')
        sys.exit(0)

# Main function for training and validating the model
def train_net(dataset_name):
    # Create training and validation datasets
    train_dataset = BasicDataset(images_dir=f'{ph.root_dir}/{dataset_name}/train/image/',
                                 labels_dir=f'{ph.root_dir}/{dataset_name}/train/label/',
                                 train=True)
    val_dataset = BasicDataset(images_dir=f'{ph.root_dir}/{dataset_name}/val/image/',
                               labels_dir=f'{ph.root_dir}/{dataset_name}/val/label/',
                               train=False)

    # Mark the size of datasets
    n_train = len(train_dataset)
    n_val = len(val_dataset)

    # Create data loaders
    loader_args = dict(num_workers=8,
                       prefetch_factor=5,
                       persistent_workers=True)
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, batch_size=ph.batch_size, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, batch_size=1, **loader_args)

    # Initialize logging
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.basicConfig(level=logging.INFO)
    localtime = time.asctime(time.localtime(time.time()))
    hyperparameter_dict = ph.state_dict()
    hyperparameter_dict['time'] = localtime
    log_wandb = wandb.init(project=ph.log_wandb_project, resume='allow', anonymous='must',
                           settings=wandb.Settings(start_method='thread'),
                           config=hyperparameter_dict, mode='offline')
    os.environ["WANDB_DIR"] = f"./{ph.log_wandb_project}"

    # Log critical configuration information
    logging.info(f'''Starting training:
        Epochs:          {ph.epochs}
        Batch size:      {ph.batch_size}
        Learning rate:   {ph.learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {ph.save_checkpoint}
        save best model: {ph.save_best_model}
        Device:          {device.type}
    ''')

    # Set up model, optimizer, warmup scheduler, learning rate scheduler, loss function, and others

    net = MambaUNet(num_classes=1, in_channels=4) # Using MambaUNet

    #net = ResUNet(num_classes=1, in_channels=4)  # Using ResUNet

    #net = TransUNet(num_classes=1, in_channels=4) # Using TransUNet

    #net = UNet(num_classes=1, in_channels=4)  # Using BaseUNet

    net = net.to(device=device)

    optimizer = optim.Adam(net.parameters(), lr=ph.learning_rate)

    grad_scaler = torch.cuda.amp.GradScaler()

    # Use cosine annealing learning rate scheduler (recommended)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ph.epochs, eta_min=ph.learning_rate * 0.01)

    # Load model and optimizer
    if ph.load:
        checkpoint = torch.load(ph.load, map_location=device)
        net.load_state_dict(checkpoint['net'])
        logging.info(f'Model loaded from {ph.load}')
        if 'optimizer' in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer'])
            for g in optimizer.param_groups:
                g['lr'] = ph.learning_rate
            optimizer.param_groups[0]['capturable'] = True

    total_step = 0
    lr = ph.learning_rate
    criterion = FCCDN_loss_without_seg
    #criterion = U2NetLoss

    # Initialize best metrics
    best_metrics = {
        'lowest_loss': float('inf'),
        'best_epoch_loss': -1,
        'highest_f1': 0.0,  # Initialize the highest F1 score to 0
        'best_epoch_f1': -1  # Initialize the best F1 score's epoch
    }
    metric_collection = MetricCollection({
        'accuracy': Accuracy(task="binary").to(device=device),
        'precision': Precision(task="binary").to(device=device),
        'recall': Recall(task="binary").to(device=device),
        'f1score': F1Score(task="binary").to(device=device),
        'miou': JaccardIndex(task="binary", num_classes=2).to(device=device)
    })

    to_pilimg = T.ToPILImage()  # Used to convert tensor (Tensor) or ndarray (NumPy array) to PIL image. Viewable in media

    # Model save paths
    checkpoint_path = f'./{ph.project_name}_checkpoint/'
    best_loss_model_path = f'./{ph.project_name}_best_loss_model/'
    best_f1_model_path = f'./{ph.project_name}_best_f1_model/'
    non_improved_epoch = 0

    for epoch in range(ph.epochs):
        print('Start Train!')

        # Training phase
        log_wandb, net, optimizer, grad_scaler, total_step, lr = train_val_test(
            mode='train', dataset_name=dataset_name,
            dataloader=train_loader, device=device, log_wandb=log_wandb, net=net,
            optimizer=optimizer, total_step=total_step, lr=lr, criterion=criterion,
            metric_collection=metric_collection, to_pilimg=to_pilimg, epoch=epoch,
            warmup_lr=None, grad_scaler=grad_scaler
        )

        # Update learning rate
        scheduler.step()


        # Validation phase
        if (epoch + 1) >= ph.evaluate_epoch and (epoch + 1) % ph.evaluate_inteval == 0:
            print('Start Validation!')

            with torch.no_grad():
                log_wandb, net, optimizer, total_step, lr, best_metrics, non_improved_epoch  = train_val_test(
                    mode='val', dataset_name=dataset_name,
                    dataloader=val_loader, device=device, log_wandb=log_wandb, net=net,
                    optimizer=optimizer, total_step=total_step, lr=lr, criterion=criterion,
                    metric_collection=metric_collection, to_pilimg=to_pilimg, epoch=epoch,
                    best_metrics=best_metrics, checkpoint_path=checkpoint_path,
                    best_loss_model_path=best_loss_model_path,best_f1_model_path=best_f1_model_path,
                    non_improved_epoch=non_improved_epoch
                )

    # Save the best model after the last training
    print(f'Best model at epoch {best_metrics["best_epoch"]} with lowest loss {best_metrics["lowest_loss"]}')
    wandb.finish()

if __name__ == '__main__':
    auto_experiment()
