import torch
import torch.nn.functional as F
from utils.utils import GreedyDecoder, cer, wer
import wandb

def test(model, device, test_loader, criterion, verbose=True):
    """
    Evaluate the performance of a model on a test dataset.
    This function evaluates a given model using a test dataset and computes
    the average loss, Character Error Rate (CER), and Word Error Rate (WER).
    It also logs these metrics using Weights and Biases (wandb).
    Args:
        model (torch.nn.Module): The model to be evaluated.
        device (torch.device): The device (CPU or GPU) to run the evaluation on.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        criterion (torch.nn.Module): Loss function used for evaluation.
        verbose (bool, optional): If True, prints detailed predictions and metrics for each sample. Defaults to True.
    Returns:
        None
    Logs:
        - TestCER: Average Character Error Rate (CER) across the test dataset.
        - TestWER: Average Word Error Rate (WER) across the test dataset.
        - TestLoss: Average loss across the test dataset.
    Prints:
        - Batch-wise predictions, targets, CER, and WER if `verbose` is True.
        - Summary of average loss, CER, and WER for the test dataset.
    """
    print('\nevaluating...')
    model.eval()
    test_loss = 0
    test_cer, test_wer = [], []
    with torch.no_grad():
        for i, _data in enumerate(test_loader):
            print(f"Batch {i}")
            spectrograms, labels, input_lengths, label_lengths = _data
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1) # (time, batch, n_class)

            loss = criterion(output, labels, input_lengths, label_lengths)
            test_loss += loss.item() / len(test_loader)

            decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1), labels, label_lengths)
            for j in range(len(decoded_preds)):
                c = cer(decoded_targets[j], decoded_preds[j])
                w = wer(decoded_targets[j], decoded_preds[j])
                if verbose:
                    print("----------------------------")
                    print(f"Predicted: {decoded_preds[j]}")
                    print(f"Target: {decoded_targets[j]}")
                    print(f"CER: {c}, WER: {w}")
                test_cer.append(c)
                test_wer.append(w)
                    


    avg_cer = sum(test_cer)/len(test_cer)
    avg_wer = sum(test_wer)/len(test_wer)
    wandb.log({"TestCER": avg_cer})
    wandb.log({"TestWER": avg_wer})
    wandb.log({"TestLoss": test_loss})

    print('Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(test_loss, avg_cer, avg_wer))