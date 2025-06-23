import numpy as np
import torch
import torchaudio
import torch.nn as nn

def avg_wer(wer_scores, combined_ref_len):
    """
    Calculate the average Word Error Rate (WER).

    This function computes the average WER by dividing the sum of WER scores 
    by the combined reference length.

    :param wer_scores: List of WER scores (numerical values).
    :type wer_scores: list[float]
    :param combined_ref_len: The total reference length used for normalization.
    :type combined_ref_len: float
    :return: The average WER as a floating-point number.
    :rtype: float
    """
    return float(sum(wer_scores)) / float(combined_ref_len)


def _levenshtein_distance(ref, hyp):
    """
    Compute the Levenshtein distance between two strings.
    The Levenshtein distance is a measure of the minimum number of single-character
    edits (insertions, deletions, or substitutions) required to change one string
    into the other. This implementation uses a space-efficient dynamic programming
    approach with a time complexity of O(m * n) and a space complexity of O(min(m, n)),
    where `m` and `n` are the lengths of the input strings.
    Parameters
    ----------
    ref : str
        The reference string.
    hyp : str
        The hypothesis string.
    Returns
    -------
    int
        The Levenshtein distance between the two input strings.
    Notes
    -----
    - If the two strings are identical, the function returns 0.
    - If one of the strings is empty, the function returns the length of the other string.
    - The function swaps the input strings if the reference string is shorter than
      the hypothesis string to optimize space usage.
    Examples
    --------
    >>> _levenshtein_distance("kitten", "sitting")
    3
    >>> _levenshtein_distance("flaw", "lawn")
    2
    >>> _levenshtein_distance("", "abc")
    3
    >>> _levenshtein_distance("abc", "abc")
    0
    """

    m = len(ref)
    n = len(hyp)

    # special case
    if ref == hyp:
        return 0
    if m == 0:
        return n
    if n == 0:
        return m

    if m < n:
        ref, hyp = hyp, ref
        m, n = n, m

    distance = np.zeros((2, n + 1), dtype=np.int32)

    for j in range(0,n + 1):
        distance[0][j] = j

    for i in range(1, m + 1):
        prev_row_idx = (i - 1) % 2
        cur_row_idx = i % 2
        distance[cur_row_idx][0] = i
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                distance[cur_row_idx][j] = distance[prev_row_idx][j - 1]
            else:
                s_num = distance[prev_row_idx][j - 1] + 1
                i_num = distance[cur_row_idx][j - 1] + 1
                d_num = distance[prev_row_idx][j] + 1
                distance[cur_row_idx][j] = min(s_num, i_num, d_num)

    return distance[m % 2][n]


def word_errors(reference, hypothesis, ignore_case=False, delimiter=' '):
    """
    Calculate the word-level edit distance (Levenshtein distance) between a reference string
    and a hypothesis string, and return the edit distance along with the number of words
    in the reference string.
    :param reference: str
        The reference string to compare against.
    :param hypothesis: str
        The hypothesis string to evaluate.
    :param ignore_case: bool, optional
        If True, the comparison will be case-insensitive. Default is False.
    :param delimiter: str, optional
        The delimiter used to split the strings into words. Default is a single space (' ').
    :returns: tuple
        A tuple containing:
        - float: The edit distance between the reference and hypothesis strings.
        - int: The number of words in the reference string.
    :raises ValueError:
        If either `reference` or `hypothesis` is not a string.
    :example:
        >>> word_errors("hello world", "hello there", ignore_case=True)
        (1.0, 2)
    """
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    ref_words = reference.split(delimiter)
    hyp_words = hypothesis.split(delimiter)

    edit_distance = _levenshtein_distance(ref_words, hyp_words)
    return float(edit_distance), len(ref_words)


def char_errors(reference, hypothesis, ignore_case=False, remove_space=False):
    """
    Calculate the character-level edit distance (Levenshtein distance) between 
    a reference string and a hypothesis string, with options to ignore case 
    and remove spaces.
    :param reference: str
        The reference string to compare against.
    :param hypothesis: str
        The hypothesis string to compare.
    :param ignore_case: bool, optional
        If True, the comparison will be case-insensitive. Default is False.
    :param remove_space: bool, optional
        If True, spaces will be removed from both strings before comparison. 
        Default is False.
    :returns: tuple
        A tuple containing:
        - float: The edit distance between the processed reference and hypothesis strings.
        - int: The length of the processed reference string.
    :rtype: tuple(float, int)
    """

    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    join_char = ' '
    if remove_space == True:
        join_char = ''

    reference = join_char.join(filter(None, reference.split(' ')))
    hypothesis = join_char.join(filter(None, hypothesis.split(' ')))

    edit_distance = _levenshtein_distance(reference, hypothesis)
    return float(edit_distance), len(reference)


def wer(reference, hypothesis, ignore_case=False, delimiter=' '):
    """
    Calculate the Word Error Rate (WER) between a reference and a hypothesis.
    WER is a common metric used to evaluate the performance of automatic speech 
    recognition systems. It is calculated as the edit distance (number of 
    insertions, deletions, and substitutions) divided by the number of words 
    in the reference.
    :param reference: str
        The reference text (ground truth).
    :param hypothesis: str
        The hypothesis text (predicted output).
    :param ignore_case: bool, optional
        If True, the comparison will be case-insensitive. Default is False.
    :param delimiter: str, optional
        The delimiter used to split the reference and hypothesis into words. 
        Default is a single space (' ').
    :raises ValueError:
        If the reference contains zero words (i.e., its length is 0 after splitting).
    :return: float
        The Word Error Rate (WER), calculated as the ratio of edit distance 
        to the number of words in the reference.
    """

    edit_distance, ref_len = word_errors(reference, hypothesis, ignore_case,
                                         delimiter)

    if ref_len == 0:
        raise ValueError("Reference's word number should be greater than 0.")

    wer = float(edit_distance) / ref_len
    return wer


def cer(reference, hypothesis, ignore_case=False, remove_space=False):
    """
    Calculate the Character Error Rate (CER) between a reference string and a hypothesis string.
    CER is a common metric for evaluating the performance of text recognition systems. 
    It is calculated as the edit distance (Levenshtein distance) between the reference 
    and hypothesis strings, divided by the length of the reference string.
    :param reference: str
        The reference string (ground truth).
    :param hypothesis: str
        The hypothesis string (predicted text).
    :param ignore_case: bool, optional
        If True, ignore case when comparing characters. Default is False.
    :param remove_space: bool, optional
        If True, remove spaces from both reference and hypothesis before comparison. Default is False.
    :raises ValueError:
        If the length of the reference string is 0.
    :return: float
        The Character Error Rate (CER), a value between 0 and 1, where 0 indicates a perfect match.
    """
    edit_distance, ref_len = char_errors(reference, hypothesis, ignore_case,
                                         remove_space)

    if ref_len == 0:
        raise ValueError("Length of reference should be greater than 0.")

    cer = float(edit_distance) / ref_len
    return cer


class TextTransform:
    """
    A utility class for transforming text to integer sequences and vice versa
    based on a predefined character mapping.
    Attributes
    ----------
    char_map : dict
        A dictionary mapping characters to their corresponding integer indices.
    index_map : dict
        A dictionary mapping integer indices back to their corresponding characters.
    Methods
    -------
    text_to_int(text)
        Converts a given text string into a sequence of integers based on the character mapping.
    int_to_text(labels)
        Converts a sequence of integers back into a text string based on the character mapping.
        Converts a given text string into a sequence of integers.
        Parameters
        ----------
        text : str
            The input text string to be converted.
        Returns
        -------
        list of int
            A list of integers representing the input text based on the character mapping.
        pass
        Converts a sequence of integers back into a text string.
        Parameters
        ----------
        labels : list of int
            A list of integers representing a sequence of characters.
        Returns
        -------
        str
            The reconstructed text string based on the integer sequence.
        pass
        """
    def __init__(self):
        char_map_str = """
        ' 0
        <SPACE> 1
        a 2
        b 3
        c 4
        d 5
        e 6
        f 7
        g 8
        h 9
        i 10
        j 11
        k 12
        l 13
        m 14
        n 15
        o 16
        p 17
        q 18
        r 19
        s 20
        t 21
        u 22
        v 23
        w 24
        x 25
        y 26
        z 27
        """
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[1] = ' '

    def text_to_int(self, text):
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char_map['<SPACE>']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string).replace('<SPACE>', ' ')

train_audio_transforms = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
    torchaudio.transforms.TimeMasking(time_mask_param=100)
)

valid_audio_transforms = torchaudio.transforms.MelSpectrogram()

text_transform = TextTransform()

def replace_accents(text):
    """
    Replace accented characters and certain special characters in a string with their unaccented or simplified equivalents.
    This function iterates through a predefined dictionary of replacements, where keys are accented or special characters
    and values are their corresponding unaccented or simplified replacements. It replaces all occurrences of these characters
    in the input text.
    :param text: str
        The input string containing accented or special characters to be replaced.
    :return: str
        A new string with all specified characters replaced by their unaccented or simplified equivalents.
    :example:
        >>> replace_accents("Héllo, wörld! Ça va?")
        'Hello world Ca va '
    """
    
    replacements = {
        'à': 'a', 'á': 'a', 'â': 'a', 'ä': 'a',
        'è': 'e', 'é': 'e', 'ê': 'e', 'ë': 'e',
        'ì': 'i', 'í': 'i', 'î': 'i', 'ï': 'i',
        'ò': 'o', 'ó': 'o', 'ô': 'o', 'ö': 'o',
        'ù': 'u', 'ú': 'u', 'û': 'u', 'ü': 'u',
        'ñ': 'n', 'ç': 'c',
        'À': 'A', 'Á': 'A', 'Â': 'A', 'Ä': 'A',
        'È': 'E', 'É': 'E', 'Ê': 'E', 'Ë': 'E',
        'Ì': 'I', 'Í': 'I', 'Î': 'I', 'Ï': 'I',
        'Ò': 'O', 'Ó': 'O', 'Ô': 'O', 'Ö': 'O',
        'Ù': 'U', 'Ú': 'U', 'Û': 'U', 'Ü': 'U',
        'Ñ': 'N', 'Ç': 'C', '.': ' ', ',': ' ',
        '!': ' ', '?': ' ', ':': ' ', ';': ' ',
        '“': ' ', '”': ' ', '‘': ' ', '’': ' ',
        '—': ' ', '–': ' ', '…': ' ', '•': ' ',
        '°': ' ', '´': ' ', '`': ' ', '^': ' ',
        '¨': ' ', '•': ' ', '§': ' ', '©': ' ',
        '"': '', '-': ' ', '·': '', '«': ' ', '»': ' ', 'ã' : 'a'
        # Add more replacements as needed
    }
    for accented_char, unaccented_char in replacements.items():
        text = text.replace(accented_char, unaccented_char)
    return text

def clean_text(text):
    """
    Cleans the input text by performing the following operations:
    - Converts all characters to lowercase.
    - Replaces accented characters with their unaccented equivalents.
    - Removes any character that is not a letter (including accented letters) or a space.

    :param text: str
        The input text to be cleaned.
    :return: str
        The cleaned text containing only lowercase alphabetic characters, accented letters, and spaces.
    """
    text = text.lower()
    text = replace_accents(text)
    for c in text:
        if c not in 'abcdefghijklmnopqrstuvwxyzáéíóúàèìòùâêîôûäëïöüñç ':
            text = text.replace(c, '')
            #print(f"Removed character: {c}")
    text = ''.join(c for c in text if c.isalpha() or c.isspace())
    return text

def data_processing(data, data_type="train"):
    """
    Processes audio data and corresponding labels for training or validation.
    Args:
        data (list): A list of tuples where each tuple contains:
            - waveform (Tensor): The audio waveform data.
            - _: Placeholder for unused data in the tuple.
            - utterance (str): The text transcription of the audio.
            - _: Placeholder for unused data in the tuple.
            - _: Placeholder for unused data in the tuple.
            - _: Placeholder for unused data in the tuple.
        data_type (str, optional): Specifies the type of data processing. 
            Must be either 'train' or 'valid'. Defaults to "train".
    Returns:
        tuple: A tuple containing:
            - spectrograms (Tensor): Padded spectrograms of the audio data 
              with shape (batch_size, 1, freq, time).
            - labels (Tensor): Padded tensor of label sequences.
            - input_lengths (list): List of input lengths for each spectrogram 
              (halved due to downsampling).
            - label_lengths (list): List of lengths for each label sequence.
    Raises:
        Exception: If `data_type` is not 'train' or 'valid'.
    Notes:
        - The function applies audio transformations based on the `data_type`.
        - Text transcriptions are cleaned and converted to integer sequences.
        - Spectrograms and labels are padded to ensure uniform batch sizes.
    """
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    for (waveform, _, utterance, _, _, _) in data:
        if data_type == 'train':
            spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        elif data_type == 'valid':
            spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        else:
            raise Exception('data_type should be train or valid')
        spectrograms.append(spec)
        utterance = clean_text(utterance) 
        label = torch.Tensor(text_transform.text_to_int(utterance.lower()))
        labels.append(label)
        input_lengths.append(spec.shape[0]//2)
        label_lengths.append(len(label))

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, input_lengths, label_lengths

def GreedyDecoder(output, labels, label_lengths, blank_label=28, collapse_repeated=True):
    """
    Decodes the output of a neural network using a greedy decoding algorithm.

    Args:
        output (torch.Tensor): The output tensor from the neural network of shape 
            (batch_size, sequence_length, num_classes), where `num_classes` includes 
            the blank label.
        labels (torch.Tensor): The ground truth labels tensor of shape 
            (batch_size, max_label_length).
        label_lengths (torch.Tensor): A tensor containing the lengths of each label 
            in the batch, of shape (batch_size,).
        blank_label (int, optional): The index of the blank label. Defaults to 28.
        collapse_repeated (bool, optional): Whether to collapse repeated characters 
            in the output. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - decodes (list of str): A list of decoded strings for each sequence in 
              the batch.
            - targets (list of str): A list of ground truth strings for each sequence 
              in the batch.

    Notes:
        - The function uses `torch.argmax` to select the most probable class at each 
          time step.
        - Repeated characters are collapsed if `collapse_repeated` is set to True.
        - The `text_transform.int_to_text` method is used to convert integer indices 
          to text for both the decoded output and the ground truth labels.
    """
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    targets = []
    for i, args in enumerate(arg_maxes):
        decode = []
        targets.append(text_transform.int_to_text(labels[i][:label_lengths[i]].tolist()))
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j -1]:
                    continue
                decode.append(index.item())
        decodes.append(text_transform.int_to_text(decode))
    return decodes, targets