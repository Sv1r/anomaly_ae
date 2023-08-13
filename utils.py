import cv2
import glob
import tqdm
import time
import torch
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

import settings

sns.set_style('darkgrid')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def data_preparation(train_path, test_path, train_size):
    """Read data from folder and create Pandas DataFrames for train/valid"""
    # Create DataFrame
    images_list = sorted(glob.glob(f'{train_path}/*'))
    df = pd.DataFrame()
    df['images'] = images_list
    # Split data on train/valid
    train, valid = train_test_split(
        df,
        train_size=train_size,
        random_state=42,
        shuffle=True
    )
    images_list_test = sorted(glob.glob(f'{test_path}/*'))
    test = pd.DataFrame()
    test['images'] = images_list_test
    return train, valid, test


class LocalDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.images_files = data['images'].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.images_files)

    def __getitem__(self, index):
        # Select on image-mask couple
        image_path = self.images_files[index]
        # Image processing
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        image = image.astype(np.uint8)
        # Augmentation
        if self.transform is not None:
            aug = self.transform(image=image)
            image = aug['image']
        return image


def show_image(dataloader):
    """Plot 6 images with applied augmentation"""
    mean = np.array(settings.MEAN)
    std = np.array(settings.STD)

    x_batch = next(iter(dataloader))

    fig, ax = plt.subplots(2, 3, figsize=(12, 9))
    for x_item, i, j in zip(x_batch, [0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]):
        image = x_item.permute(1, 2, 0).numpy()
        image = std * image + mean

        ax[i, j].imshow(image.clip(0, 1))
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])

    fig.tight_layout()
    plt.show()


def init_weights(m):
    try:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    except Exception:
        pass


def train_model(
        folder_to_save_weights,
        path_to_train,
        model,
        train_dataloader,
        valid_dataloader,
        test_dataloader,
        loss,
        optimizer,
        num_epoch,
        scheduler=None,
        avg_precision=True,
):
    """Model training function"""
    train_loss_history, valid_loss_history, test_loss_history = [], [], []
    # Dataframe
    df = pd.DataFrame()
    # Model to device
    model = model.to(device)
    # Scaler for average precision training
    scaler = torch.cuda.amp.GradScaler(enabled=avg_precision)
    # Initial minimum loss
    valid_min_loss = 1e3

    for epoch in range(num_epoch):
        # Each epoch has a training and validation phase
        for phase in ['Train', 'Valid', 'Test']:
            if phase == 'Train':
                dataloader = train_dataloader
                model.train()  # Set model to training mode
            elif phase == 'Valid':
                dataloader = valid_dataloader
                model.eval()  # Set model to evaluate mode
            else:
                dataloader = test_dataloader
                model.eval()  # Set model to evaluate mode
            running_loss = []
            # Iterate over data.
            with tqdm.tqdm(dataloader, unit='batch') as tqdm_loader:
                for inputs in tqdm_loader:
                    tqdm_loader.set_description(f'Epoch {epoch}/{num_epoch-1} - {phase}')
                    inputs = inputs.to(device)
                    optimizer.zero_grad()
                    # forward and backward
                    with torch.set_grad_enabled(phase == 'Train'):
                        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=avg_precision):
                            predict = model(inputs)
                            loss_value = loss(predict, inputs)
                        # backward + optimize only if in training phase
                        if phase == 'Train':
                            scaler.scale(loss_value).backward()
                            scaler.step(optimizer)
                            scaler.update()
                            if scheduler is not None:
                                scheduler.step()
                    # statistics
                    loss_item = loss_value.item()
                    running_loss.append(loss_item)
                    # Current statistics
                    tqdm_loader.set_postfix(Loss=loss_item)
                    time.sleep(.1)
            epoch_loss = np.mean(running_loss)
            # Checkpoint
            if epoch_loss < valid_min_loss and phase != 'Train':
                valid_min_loss = epoch_loss
                model = model.cpu()
                torch.save(model, f'{folder_to_save_weights}/best.pt')
                model = model.to(device)
            # Loss history
            if phase == 'Train':
                train_loss_history.append(epoch_loss)
            elif phase == 'Valid':
                valid_loss_history.append(epoch_loss)
            else:
                test_loss_history.append(epoch_loss)
            print(
                'Epoch: {}/{}  Stage: {} Loss: {:.6f}'.format(
                    epoch, num_epoch-1, phase, epoch_loss
                ), flush=True
            )
            time.sleep(.1)

    # Add results for each model
    df['Train_Loss'] = train_loss_history
    df['Valid_Loss'] = valid_loss_history
    df['Test_Loss'] = test_loss_history
    # Save df if csv format
    df.to_csv(f'{path_to_train}/results.csv', sep=' ', index=False)
    # Save last model
    torch.save(model, f'{folder_to_save_weights}/last.pt')

    return model, df


def result_plot(data, path_to_train):
    """Plot loss function and Metrics"""
    stage_list = np.unique(list(map(lambda x: x.split(sep='_')[0], data.columns)))
    variable_list = np.unique(list(map(lambda x: x.split(sep='_')[1], data.columns)))
    plt.subplots(figsize=(10, 10))
    for stage in stage_list:
        for variable in variable_list:
            plt.plot(data[f'{stage}_{variable}'], label=f'{stage}')
            plt.title(f'{variable} Plot', fontsize=10)
            plt.xlabel('Epoch', fontsize=8)
            plt.ylabel(f'{variable} Value', fontsize=8)
            plt.legend()
    plt.savefig(f'{path_to_train}/loss_plot.png')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--random-state',
        type=int,
        default=settings.RANDOM_STATE,
        help='Fix random state'
    )
    parser.add_argument(
        '--train-path',
        type=str,
        default=f'{settings.DATA_FOLDER}/ne_proliv',
        help='Path to train/val set'
    )
    parser.add_argument(
        '--test-path',
        type=str,
        default=f'{settings.DATA_FOLDER}/proliv',
        help='Path to test set'
    )
    parser.add_argument(
        '--train-size',
        type=float, default=.7,
        help='Train ratio for train_test_split'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-2,
        help='Learning rate for optimizer'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0,
        help='Weight decay for L2 regularization'
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=settings.EPOCHS,
        help='Number of epoch'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=settings.BATCH_SIZE,
        help='Batch size, in case'
    )
    parser.add_argument(
        '--use-avg-precision',
        default=True,
        action=argparse.BooleanOptionalAction,
        help='Boolean flag for average precision train mode'
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=settings.IMAGE_SIZE,
        help='Image size for train'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='exp',
        help='Name for current train'
    )

    return parser.parse_args()
