import torch
import torch.nn as nn
import torch.nn.functional as F

EMBEDDING_DIM = 256
HIDDEN_SIZE = 512
NUM_CNN_LAYERS = 3
CNN_KERNEL_SIZE = 3
CNN_STRIDE = 1
NUM_RNN_LAYERS = 2
DROPOUT = 0.1



#IMPORTANTE:
# Cuando escribi este codigo solo dios y yo sabiamos como funcionaba
# ahora solo dios lo sabe
# No toqueis nada a menos que esteis seguros de lo que haceis
# porque esto se rompe muy facil y no tengo ganas de pasarme otra noche
# entera arreglandolo.
# Tened especial cuidado con el forward del Seq2Seq, que si tocas lo mas
# minimo los tensores no tienen el mismo tamaño y se rompe todo.

class CNNEncoder(nn.Module):
    """
    Codificador basado en CNN que transforma imágenes en secuencias de vectores.

    Args:
        input_channels (int): Número de canales de entrada (por defecto 1 para imágenes en escala de grises).
        n_layers (int): Número de capas convolucionales.
        kernel_size (int): Tamaño del kernel de las convoluciones.
        stride (int): Stride de las convoluciones.
        output_size (int): Tamaño base de los mapas de características generados.

    Métodos:
        forward(x): Procesa el tensor de entrada x y devuelve una secuencia (batch, width, features).
    """
    def __init__(self, input_channels = 1, n_layers = NUM_CNN_LAYERS, kernel_size = CNN_KERNEL_SIZE, stride = CNN_STRIDE, output_size = 32):
        super().__init__()
        layers = []
        input_size = input_channels
        for i in range(n_layers):
            conv = nn.Conv2d(input_size, output_size * (2**i), kernel_size = kernel_size, stride = stride, padding = (kernel_size // 2) )
            relu = nn.ReLU()
            bn = nn.BatchNorm2d(output_size * (2**i))
            layers.append(nn.Sequential(conv, bn, relu))
            input_size = output_size * (2**i)
        
        self.cnn = nn.Sequential(*layers)
        
    
    def forward(self, x):
        out = self.cnn(x)
        batch_size, channels, height, width = out.size()
        out = out.permute(0, 3, 1, 2).contiguous().view(
            batch_size,
            width,
            channels * height
        )

        return out


class CNNEncoder1d(nn.Module):
    """
    Codificador basado en CNN que transforma imágenes en secuencias de vectores.

    Args:
        input_channels (int): Número de canales de entrada (por defecto 1 para imágenes en escala de grises).
        n_layers (int): Número de capas convolucionales.
        kernel_size (int): Tamaño del kernel de las convoluciones.
        stride (int): Stride de las convoluciones.
        output_size (int): Tamaño base de los mapas de características generados.

    Métodos:
        forward(x): Procesa el tensor de entrada x y devuelve una secuencia (batch, width, features).
    """
    
    def __init__(self, input_channels=1, n_layers=NUM_CNN_LAYERS, kernel_size=5, stride=2, output_size=64):
        super().__init__()
        layers = []
        self.strides = []
        in_channels = input_channels
        for i in range(n_layers):
            out_channels = output_size * (2**i)
            conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
            bn = nn.BatchNorm1d(out_channels)
            relu = nn.ReLU()
            layers.append(nn.Sequential(conv, bn, relu))
            in_channels = out_channels
            self.strides.append(stride) # Poder recuperar el output length
        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, 1, T)
        out = self.cnn(x)  # -> (B, C, T')
        out = out.permute(0, 2, 1)  # -> (B, T', C)
        return out

    def get_output_lengths(self, input_lengths):
        for layer in self.cnn:
            conv = layer[0]
            kernel_size = conv.kernel_size[0]
            stride = conv.stride[0]
            padding = conv.padding[0]
            input_lengths = ((input_lengths + 2 * padding - kernel_size) // stride) + 1
        return input_lengths
                     
class RNNEncoder(nn.Module):
    """
    Codificador basado en LSTM que procesa secuencias obtenidas del codificador CNN.

    Args:
        input_size (int): Tamaño del vector de entrada.
        hidden_size (int): Tamaño del estado oculto.
        n_layers (int): Número de capas LSTM.
        dropout (float): Dropout entre capas LSTM.
        bidirectional (bool): Si el LSTM es bidireccional o no.

    Métodos:
        forward(x, hidden): Procesa la secuencia x con estado oculto inicial.
        init_hidden(batch_size, device): Inicializa los estados oculto y de celda.
    """
    def __init__(self, input_size, hidden_size, n_layers = NUM_RNN_LAYERS, dropout = DROPOUT, bidirectional = True):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, n_layers, dropout = dropout, bidirectional = bidirectional, batch_first = True)
        self.hidden_size = hidden_size
        
    def forward(self, x, hidden = None):
        out, _ = self.rnn(x, hidden)
        return out, hidden
    
    def init_hidden(self, batch_size, device):
        num_directions = 2 if self.rnn.bidirectional else 1 
        h0 = torch.zeros(self.rnn.num_layers * num_directions, batch_size, self.rnn.hidden_size).to(device)
        c0 = torch.zeros(self.rnn.num_layers * num_directions, batch_size, self.rnn.hidden_size).to(device)
        
        return (h0, c0)
    
class Attention(nn.Module):
    """
    Módulo de atención que calcula pesos de atención sobre las salidas del codificador.

    Args:
        enc_hidden_size (int): Tamaño del estado oculto del codificador.
        dec_hidden_size (int): Tamaño del estado oculto del decodificador.

    Métodos:
        forward(hidden, encoder_outputs): Calcula pesos de atención entre el decodificador y la salida del codificador.
    """
    def __init__(self, enc_hidden_size, dec_hidden_size):
        super().__init__()
        self.attn = nn.Linear(enc_hidden_size * 2 + dec_hidden_size, dec_hidden_size )
        self.v = nn.Linear(dec_hidden_size, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
        batch_size, seq_len, _ = encoder_outputs.size()
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        attn_input = torch.cat((hidden, encoder_outputs), dim=2)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2)))
        attention = self.v(energy).squeeze(2)
        
        return torch.softmax(attention, dim = 1).unsqueeze(1)
       
class RNNDecoder(nn.Module):
    """
    Decodificador con atención basado en LSTM para generar secuencias de salida.

    Args:
        output_size (int): Tamaño del vocabulario de salida.
        embedding_dim (int): Tamaño del vector de embedding.
        hidden_size (int): Tamaño del estado oculto.
        n_layers (int): Número de capas LSTM.
        dropout (float): Dropout entre capas LSTM.
        bidirectional_encoder (bool): Si el codificador fue bidireccional.

    Métodos:
        forward(inp, hidden, encoder_outputs): Decodifica un paso de la secuencia.
        init_hidden(encoder_hidden): Inicializa el estado del decodificador a partir del codificador.
    """
    def __init__(self, output_size, embedding_dim, hidden_size, n_layers = NUM_RNN_LAYERS, dropout = DROPOUT, bidirectional_encoder = True):
        super().__init__()
        self.embedding = nn.Embedding(output_size, embedding_dim)
        self.hidden_size = hidden_size
        self.num_layers = n_layers
        self.attention = Attention(
            hidden_size, 
            hidden_size
        )
        
        self.rnn = nn.LSTM(
            embedding_dim + (hidden_size * 2 if bidirectional_encoder else hidden_size),
            hidden_size,
            n_layers,
            dropout = dropout,
            batch_first = True
        )
        
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
    
    def forward(self, inp, hidden, encoder_outputs):
        
        hidden, cell_state = hidden
        embedded = self.dropout(self.embedding(inp))

        attn_weights = self.attention(hidden[-1], encoder_outputs)
        context = torch.bmm(attn_weights, encoder_outputs)

        rnn_input = torch.cat((embedded, context), dim = 2)

        output, (hidden, cell_state) = self.rnn(rnn_input, (hidden, cell_state))
        prediction = self.out(output.squeeze(1))
        return prediction, (hidden, cell_state), attn_weights
    
    def init_hidden(self, encoder_hidden):
        return encoder_hidden[:self.num_layers]
    

class Seq2Seq(nn.Module):
    def __init__(self, encoder_cnn, encoder_rnn, decoder_rnn, device):
        super().__init__()
        self.encoder_cnn = encoder_cnn
        self.encoder_rnn = encoder_rnn
        self.decoder = decoder_rnn
        self.device = device
        self.subi_putero = None
    
    def forward(self, src, trg, teacher_forcing_ratio = .5):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.out.out_features
        
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        cnn_outputs = self.encoder_cnn(src)
        encoder_hidden = self.encoder_rnn.init_hidden(batch_size, self.device)
        encoder_outputs, encoder_hidden = self.encoder_rnn(cnn_outputs, encoder_hidden)

        decoder_input = trg[:, 0].unsqueeze(1)
        decoder_hidden = encoder_hidden[0][:self.decoder.num_layers], encoder_hidden[1][:self.decoder.num_layers]
        
        for t in range(1, trg_len):
            decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            outputs[t] = decoder_output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = decoder_output.argmax(1).unsqueeze(1)
            decoder_input = trg[:, t].unsqueeze(1) if teacher_force else top1
        
        return outputs
         

class CTCModel(nn.Module):
    def __init__(self, cnn_encoder, rnn_encoder, output_dim):
        super().__init__()
        self.cnn_encoder = cnn_encoder
        self.rnn_encoder = rnn_encoder
        # No tiene decoder porque la CTC no lo necesita
        self.fc_out = nn.Linear(rnn_encoder.hidden_size * 2, output_dim)  # *2 if bidirectional

    def forward(self, src):
        conv_out = self.cnn_encoder(src)          # (N, T, F)
        
        rnn_out, _ = self.rnn_encoder(conv_out)   # (N, T, H)
        logits = self.fc_out(rnn_out)             # (N, T, VOCAB_SIZE)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.permute(1, 0, 2)  # Devuelve (T, N, VOCAB_SIZE) para CTC