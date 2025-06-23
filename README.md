![image](/resources/CatSpeech.png)

This repository consists on a Deep Learning project whose main goal is to correctly transcribe catalan audios into text using an Automatic Speech Recognition model. The data used to train and test the model is [Mozilla's Catalan Commonvoice (16.0) ](https://huggingface.co/datasets/mozilla-foundation/common_voice_16_0).

## Table of Contents
 - [Project Structure](#Project-structure)
 - [Data](#Data)
 - [Overview](#overview)
 - [How to run](#How-to-run)

## Project Structure
- `data/`: Contains the `.parquet` files of the dataset.
- `models/`: Contains the models architecture code.
- `utils/`: Contains support functions and data preprocessing.
- `information/`: Contains markdown files with information about architecture and results.
- `docs/`: Contains the scripts necessary to build the documentation with sphinx along with the actual documentation
    - The documentation can be read running `firefox docs/_build/html/index.html` or substituting firefox by the desired browser
- `run.py`: Script to run the project.
- `test.py`: Function to test the model.
- `train.py`: Function to train one epoch.


## Data
As stated earlier the used dataset is the mozilla catalan commonvoice, the dataset consists on a parquet file that links a `.tsv` with the audio directory, the `.tsv` file contains the following metadata:
| client_id        | path                  | sentence                             | up_votes | down_votes | age   | gender | accent     |
|-------------------|-----------------------|--------------------------------------|----------|------------|-------|--------|------------|
| 9987d47b8c6b6c3b | clips/common_voice.wav | Hola, com estàs?                     | 2        | 0          | 30-39 | male   | Balear    |
| e8c4f79dcb6f6acb | clips/common_voice.wav | Bon dia, avui fa molt bon temps.     | 3        | 1          | 40-49 | female | Central   |
| c3d5e78dcb4e4a4c | clips/common_voice.wav | Aquest és un exemple de registre.    | 1        | 0          | 20-29 | male   |  Rosellonés|

## Overview
In this project we tackle the problem of building an automatic speech recognition model to transcribe catalan audios into text this is done by:

#### Data preprocessing
Audios:
 - The audios are loaded as waveforms using the `torchaudio` library.
 - The sample rate is standarized to 16KHz.
 - Transforms are applied.
    - Train audios: MelSpectrograms, FrequencyMasking and TimeMasking.
    - Validation audios: MelSpectrograms.
 - The spectrograms are padded to standarize length.

Text:
 - Transcripts at tokenized at character level and cleaned of unusual characters.
 - Characters are encoded to integers.

#### Model outline
The model consist on a residual CNN for feature extraction, then an RNN consisting on Bidirectional GRUs for sequence interpretation and lastly a fully-connected layer to classify into characters. This is built on a CTC loss and then decoded using a greedy approach. 

For more information about the architecture read [Architecture](/information/architecture.md).

#### Results

After training for 10 epochs the results obtained are:

| Mean Train Loss | Mean Validation Loss | Mean WER | Mean CER | 
| --------------- | -------------------- | -------- | -------- | 
| 0.61      |0.51           |0.48|0.16|

For a more thorough result analysis read [Results](/information/results.md)
## How to run
1. Clone the repository:
```
git clone https://github.com/DCC-UAB/24-25-xnap-projecte-matcad_grup_5.git
```

2. Install the dependencies:

We strongly recommend creating a conda environment with the provided `environment.yml` file

```
conda env create -f environment.yml
conda activate speech2text
```

However it is possible to install everything with:

```
pip install -r requirements.txt
```

3. Get the data

If you wish to train with the mozilla commonvoice 16.0 data as we did you can download the dataset with:

```
python load_data.py
```
Note that in order to download this dataset you will need to have your huggingface api key stored as `HUGGINFACE_API_KEY`

****Warning: mozilla's catalan commonvoice 16 dataset is over 75 GB**


4. Run the project

To train a model using the loaded data you must run:
```
python run.py <flags>
```
The script accepts multiple flags:

**Parameter flags**

By default the model runs on its default parameters however any of these can be overriden using the following flags:
```
--learning_rate
--batch_size
--epochs
--n_cnn_layers
--n_rnn_layers
--rnn_dim
--n_class
--n_feats
--stride
--dropout
```
**Warning: Some of this parameters (for instance `n_class`) are directly linked to the data encoding or model architecture, incorrect modification can lead to the code not running**

These parameters can be also overriden all at once using the `--hparams` flag, providing a JSON-like string, for instance:
```
python run.py --hparams {
        "n_cnn_layers": 3,     \
        "n_rnn_layers": 5,     \
        "rnn_dim": 512,        \
        "n_class": 29,         \
        "n_feats": 128,        \
        "stride":2,            \
        "dropout": 0.1,        \
        "learning_rate": 5e-4, \
        "batch_size": 6,       \
        "epochs": 10}
```

When the model is trained it is possible to test using
```
python run.py --test
```

Complimentary to this script two flags are available:

`--model-path <path>` Specifies the path of the saved model to load.

`--verbose_test` If present makes the test output display the predicted and target transcriptions and the WER and CER score of each transcription.

## Contributors
Samuel Ortega: Samuel.OrtegaC@autonoma.cat

Nicolás Romeu: Nicolas.Romeu@autonoma.cat

David Ruiz: David.RuizCac@autonoma.cat

Marc Roig: Marc.RoigO@autonoma.cat


Neural Networks and Deep Learning


Degree in Computational Mathematics & Data analyitics
UAB, 2025
