import os
import torch
import random
import numpy as np

import transforms
import utils
import model_architecture


def main(opt):
    # Get opt arguments
    random_state = opt.random_state
    train_path = opt.train_path
    test_path = opt.test_path
    train_size = opt.train_size
    learning_rate = opt.learning_rate
    weight_decay = opt.weight_decay
    epoch = opt.epoch
    batch_size = opt.batch_size
    use_avg_precision = opt.use_avg_precision
    image_size = opt.image_size
    name = opt.name

    # Fix random behavior
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)
    # Get train/valid DataFrames
    train, valid, test = utils.data_preparation(train_path=train_path, test_path=test_path, train_size=train_size)
    # Augmentation
    train_aug = transforms.get_train_aug(image_size=image_size)
    valid_aug = transforms.get_valid_aug(image_size=image_size)
    # Datasets
    train_dataset = utils.LocalDataset(
        train,
        transform=train_aug
    )
    valid_dataset = utils.LocalDataset(
        valid,
        transform=valid_aug
    )
    test_dataset = utils.LocalDataset(
        test,
        transform=valid_aug
    )
    # Dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True)
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )
    # Loss function
    loss = torch.nn.MSELoss()
    # Model
    model = model_architecture.Anomaly()
    # Initialize model weights
    model.apply(utils.init_weights)
    # Optimizer with weight_decay
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Scheduler
    scheduler = None
    # Save results folders
    path_to_save = './runs'
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    path_to_train = f'{path_to_save}/{name}'
    if not os.path.exists(path_to_train):
        folder_to_save_weights = f'{path_to_train}/weights'
        for direction in [path_to_train, folder_to_save_weights]:
            os.makedirs(direction)
    else:
        count_folders = 1
        while os.path.exists(path_to_train + f'{count_folders}'):
            count_folders += 1
        path_to_train += f'{count_folders}'
        folder_to_save_weights = f'{path_to_train}/weights'
        for direction in [path_to_train, folder_to_save_weights]:
            os.makedirs(direction)
    # Train function
    model, df = utils.train_model(
        folder_to_save_weights=folder_to_save_weights,
        path_to_train=path_to_train,
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        test_dataloader=test_dataloader,
        loss=loss,
        optimizer=optimizer,
        num_epoch=epoch,
        scheduler=scheduler,
        avg_precision=use_avg_precision
    )
    # Plot results
    utils.result_plot(data=df, path_to_train=path_to_train)


if __name__ == '__main__':
    opt_input = utils.parse_opt()
    main(opt=opt_input)
