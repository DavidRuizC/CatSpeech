import os
import torchaudio
from torch.utils.data import Dataset
import pandas as pd

class CommonVoiceCatalanParquetDataset(Dataset):
    """
    A PyTorch Dataset for loading and processing the Common Voice Catalan dataset stored in Parquet format.
    Attributes:
        root_dir (str): The root directory where the dataset is stored. Defaults to './data/cv_ca_'.
        audio_dir (str): The directory containing audio clips, derived from `root_dir`.
        split (str): The dataset split to use (e.g., "train", "test", "validation"). Defaults to "train".
        transform (callable, optional): A function/transform to apply to the audio waveform. Defaults to None.
        metadata (pd.DataFrame): A DataFrame containing metadata loaded from the Parquet file.
    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Retrieves the sample at the specified index.
    Args:
        root_dir (str, optional): The root directory where the dataset is stored. Defaults to './data/cv_ca_'.
        split (str, optional): The dataset split to use (e.g., "train", "test", "validation"). Defaults to "train".
        transform (callable, optional): A function/transform to apply to the audio waveform. Defaults to None.
    Example:
        >>> dataset = CommonVoiceCatalanParquetDataset(root_dir='./data/cv_ca_', split='train')
        >>> print(len(dataset))
        >>> waveform, sample_rate, sentence, client_id, _, _ = dataset[0]
    """
    def __init__(self, root_dir='./data/cv_ca_', split="train", transform=None):
        self.root_dir = root_dir
        self.audio_dir = os.path.join(root_dir, "clips")
        self.split = split
        self.transform = transform

        parquet_path = root_dir + f"{split}.parquet"
        print(parquet_path)
        self.metadata = pd.read_parquet(parquet_path)[:100000]
        print(f"Training set size: {len(self.metadata)}")
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        audio_path = os.path.join(self.audio_dir, row["path"])  # path must be the filename column
        waveform, sample_rate = torchaudio.load(audio_path)

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, sample_rate, row["sentence"], row["client_id"], 0, 0 #Testing only