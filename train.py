import wandb
import torch.nn.functional as F

def train(model, device, train_loader, criterion, optimizer, scheduler, epoch):
    """
    Train the model for one epoch.
    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to be trained.
    device : torch.device
        The device (CPU or GPU) on which the model and data are located.
    train_loader : torch.utils.data.DataLoader
        DataLoader providing the training dataset in batches.
    criterion : callable
        The loss function used to compute the error between predictions and targets.
    optimizer : torch.optim.Optimizer
        The optimization algorithm used to update model parameters.
    scheduler : torch.optim.lr_scheduler._LRScheduler
        Learning rate scheduler to adjust the learning rate during training.
    epoch : int
        The current epoch number.
    Notes
    -----
    - The function sets the model to training mode.
    - It iterates over the training dataset, computes the loss, performs backpropagation, and updates the model parameters.
    - Logs the training loss to the console and to Weights & Biases (wandb) every `PRINT_EVERY` iterations.
    """
    
    PRINT_EVERY = 100
    model.train()
    data_len = len(train_loader.dataset)

    for batch_idx, _data in enumerate(train_loader):
        spectrograms, labels, input_lengths, label_lengths = _data
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        optimizer.zero_grad()

        output = model(spectrograms)  # (batch, time, n_class)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1) # (time, batch, n_class)

        loss = criterion(output, labels, input_lengths, label_lengths)
        loss.backward()

        optimizer.step()
        scheduler.step()
        if batch_idx % PRINT_EVERY == 0 or batch_idx == data_len:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(spectrograms), data_len,
                100. * batch_idx / len(train_loader), loss.item()))
            wandb.log({"TrainLoss": loss.item()})