import torch.nn as nn
import torch.nn.functional as F

class CNNLayerNorm(nn.Module):
    """
    Applies Layer Normalization over the input tensor for a Convolutional Neural Network (CNN).
    This module normalizes the input tensor along the feature dimension, ensuring that the
    mean and variance of the features are consistent across the batch. The input tensor is
    expected to have the shape (batch, channel, feature, time).
    Args:
        n_feats (int): The number of features in the input tensor.
    Methods:
        forward(x):
            Applies layer normalization to the input tensor.
            Args:
                x (torch.Tensor): Input tensor of shape (batch, channel, feature, time).
            Returns:
                torch.Tensor: Normalized tensor of the same shape as the input.
    """
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous() # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous() # (batch, channel, feature, time)

class ResidualCNN(nn.Module):
    """
    A Residual Convolutional Neural Network (ResidualCNN) module.
    This module implements a residual block with two convolutional layers, 
    dropout, layer normalization, and GELU activation. The input is added 
    to the output of the second convolutional layer to form the residual 
    connection.
    Parameters
    ----------
    in_channels : int
        Number of input channels for the first convolutional layer.
    out_channels : int
        Number of output channels for the convolutional layers.
    kernel : int
        Size of the convolutional kernel.
    stride : int
        Stride of the convolutional layers.
    dropout : float
        Dropout probability applied after each convolutional layer.
    n_feats : int
        Number of features for the layer normalization.
    Methods
    -------
    forward(x)
        Defines the forward pass of the ResidualCNN module.
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, channel, feature, time).
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, channel, feature, time).
    """

    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x # (batch, channel, feature, time)

class BidirectionalGRU(nn.Module):
    """
    A Bidirectional Gated Recurrent Unit (GRU) layer with Layer Normalization, 
    GELU activation, and Dropout.
    Parameters
    ----------
    rnn_dim : int
        The number of expected features in the input (input size of the GRU).
    hidden_size : int
        The number of features in the hidden state (hidden size of the GRU).
    dropout : float
        The dropout probability to apply after the GRU layer.
    batch_first : bool
        If True, the input and output tensors are provided as (batch, seq, feature).
        Otherwise, they are provided as (seq, batch, feature).
    Methods
    -------
    forward(x)
        Applies the Bidirectional GRU layer, followed by Layer Normalization, 
        GELU activation, and Dropout to the input tensor.
    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (batch, seq, feature) if `batch_first` is True, 
        otherwise (seq, batch, feature).
    Returns
    -------
    torch.Tensor
        Output tensor after applying the Bidirectional GRU, Layer Normalization, 
        GELU activation, and Dropout.
    """

    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x


class SpeechRecognitionModel(nn.Module):
    """
    A Speech Recognition Model that combines Convolutional Neural Networks (CNNs),
    Residual CNNs, Bidirectional Gated Recurrent Units (BiGRUs), and a classifier
    for end-to-end speech recognition.
    Parameters
    ----------
    n_cnn_layers : int
        Number of residual CNN layers to extract hierarchical features.
    n_rnn_layers : int
        Number of bidirectional GRU layers for sequential modeling.
    rnn_dim : int
        Dimension of the GRU hidden state.
    n_class : int
        Number of output classes (e.g., vocabulary size).
    n_feats : int
        Number of input features (e.g., Mel-frequency cepstral coefficients).
    stride : int, optional
        Stride for the initial CNN layer. Default is 2.
    dropout : float, optional
        Dropout probability for regularization. Default is 0.1.
    Attributes
    ----------
    cnn : nn.Conv2d
        Initial convolutional layer for feature extraction.
    rescnn_layers : nn.Sequential
        Sequence of residual CNN layers for hierarchical feature extraction.
    fully_connected : nn.Linear
        Fully connected layer to project features to the GRU input dimension.
    birnn_layers : nn.Sequential
        Sequence of bidirectional GRU layers for sequential modeling.
    classifier : nn.Sequential
        Fully connected layers and activation functions for classification.
    Methods
    -------
    forward(x)
        Defines the forward pass of the model. Takes input tensor `x` and
        returns the output tensor after passing through the CNN, residual CNNs,
        GRUs, and classifier.
    """

    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1):
        super(SpeechRecognitionModel, self).__init__()
        n_feats = n_feats//2
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3//2)  # cnn for extracting heirachal features

        # n residual cnn layers with filter size of 32
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats)
            for _ in range(n_cnn_layers)
        ])
        self.fully_connected = nn.Linear(n_feats*32, rnn_dim)
        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i==0 else rnn_dim*2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i==0)
            for i in range(n_rnn_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2) # (batch, time, feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return x