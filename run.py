import os
import json
import argparse
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from utils.utils import data_processing
from train import train
from test import test
from models.CTC import SpeechRecognitionModel
from utils.dataset import CommonVoiceCatalanParquetDataset
import warnings
warnings.filterwarnings("ignore")

def main(hparams, mode = 'train', verbose_test = False, model_path = 'models/model_epoch_9.pt'):

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(7)
    device = torch.device("cuda" if use_cuda else "cpu")

    if not os.path.isdir("./data"):
        os.makedirs("./data")
    
    from torch.utils.data import DataLoader

    train_dataset = CommonVoiceCatalanParquetDataset("./data/cv_ca_", split="train")
    test_dataset = CommonVoiceCatalanParquetDataset("./data/cv_ca_", split="validation")


    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = DataLoader(dataset=train_dataset,
                                batch_size=hparams['batch_size'],
                                shuffle=True,
                                collate_fn=lambda x: data_processing(x, 'train'),
                                **kwargs)
    test_loader = DataLoader(dataset=test_dataset,
                                batch_size=hparams['batch_size'],
                                shuffle=False,
                                collate_fn=lambda x: data_processing(x, 'valid'),
                                **kwargs)

    model = SpeechRecognitionModel(
        hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
        hparams['n_class'], hparams['n_feats'], hparams['stride'], hparams['dropout']
        ).to(device)

    print(model)
    print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))

    optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'])
    criterion = nn.CTCLoss(blank=28).to(device)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['learning_rate'],
                                            steps_per_epoch=int(len(train_loader)),
                                            epochs=hparams['epochs'],
                                            anneal_strategy='linear')

    if mode == 'train':
        wandb.login()
        wandb.init(project="CatalanSpeech2text")
        wandb.watch(model, criterion, log="all", log_freq = 10)
        SAVE_EVERY = 1
        for epoch in range(1, hparams['epochs'] + 1):
            train(model, device, train_loader, criterion, optimizer, scheduler, epoch)
            test(model, device, test_loader, criterion, False)
            if epoch % SAVE_EVERY == 0:
                torch.save(model.state_dict(), f"models/model_epoch_{epoch}.pt")
                print(f"Model saved at epoch {epoch}")
    else:
        model.load_state_dict(torch.load(model_path, map_location=device))
        test(model, device, test_loader, criterion, verbose_test)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action = "store_true", help="Mode to run the script in: 'train' or 'test'")
    parser.add_argument("--model-path", type = str, default = "models/model_epoch_9.pt", help="Path to the model file for testing")
    parser.add_argument("--verbose_test", action="store_true", help="If set, will print detailed test results")
    parser.add_argument("--hparams", type=str, default = None, help="JSON string of hyperparameters ")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate (default: 5e-4)")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size (default: 20)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs (default: 10)")
    parser.add_argument("--n_cnn_layers", type=int, default=3, help="Number of CNN layers (default: 3)")
    parser.add_argument("--n_rnn_layers", type=int, default=5, help="Number of RNN layers (default: 5)")
    parser.add_argument("--rnn_dim", type=int, default=512, help="RNN dimension (default: 512)")
    parser.add_argument("--n_class", type=int, default=29, help="Number of classes (default: 29)")
    parser.add_argument("--n_feats", type=int, default=128, help="Number of features (default: 128)")
    parser.add_argument("--stride", type=int, default=2, help="Stride (default: 2)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate (default: 0.1)")

    args = parser.parse_args()

    
    if args.hparams:
        hparams = json.loads(args.hparams)
    else:
        hparams = {
            "n_cnn_layers": args.n_cnn_layers,
            "n_rnn_layers": args.n_rnn_layers,
            "rnn_dim": args.rnn_dim,
            "n_class": args.n_class,
            "n_feats": args.n_feats,
            "stride":args.stride,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs
        }
        
    print("Using hyperparameters:")
    for k, v in hparams.items():
        print(f"  {k}: {v}")

    mode = 'train' if not args.test else 'test'
    main(hparams, mode, args.verbose_test, model_path=args.model_path)

