import torch
import torch.nn as nn
import torchaudio


# Definim el nostre vocabulari
vocab_list = list(
    "abcdefghijklmnopqrstuvwxyzçàèéíïòóúü" +
    "ABCDEFGHIJKLMNOPQRSTUVWXYZÇÀÈÉÍÏÒÓÚÜ" +
    " ,.;·'-?!\""
)

class TextTransform:
    """Maps characters to integers and vice versa"""
    def __init__(self):
        # Creem els diccionaris de mapeig
        self.char_map = {ch: i+2 for i, ch in enumerate(vocab_list)}  # Comencem des de 2 (0 i 1 estan reservats)
        self.char_map[''] = 1  # Espai en blanc (carcters no reconeguts)
        self.char_map["$"] = 0  # (blank)
        self.index_map = {i: ch for ch, i in self.char_map.items()}
        self.index_map[1] = ' '
        self.index_map[0] = "$"  # (blank)

    def text_to_int(self, text):
        """Convert text to an integer sequence using character map"""
        int_sequence = []
        for c in text:
            if c == ' ':
                int_sequence.append(self.char_map[''])
            elif c in self.char_map:
                int_sequence.append(self.char_map[c])
            else:
                # Si el caràcter no està al vocabulari, el substituïm per espai
                int_sequence.append(self.char_map[''])
        return int_sequence

    def int_to_text(self, labels):
        """Convert integer labels to text using character map"""
        string = []
        for i in labels:
            if i in self.index_map:
                string.append(self.index_map[i])
            else:
                # Si l'índex no existeix, afegim un espai
                string.append(' ')
        return ''.join(string)

# Audio transforms (igual que abans)
train_audio_transforms = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
    torchaudio.transforms.TimeMasking(time_mask_param=35)
)

valid_audio_transforms = torchaudio.transforms.MelSpectrogram()

text_transform = TextTransform()

def data_processing(data, data_type="train"):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    
    for (waveform, utterance) in data:
        if data_type == 'train':
            spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        else:
            spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
            
        spectrograms.append(spec)
        label = torch.Tensor(text_transform.text_to_int(utterance)) 
        labels.append(label)
        input_lengths.append(spec.shape[0]//2)
        label_lengths.append(len(label))
    
    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    
    return spectrograms, labels, input_lengths, label_lengths


def GreedyDecoder(output, labels, label_lengths, blank_label=0, collapse_repeated=True):
   """
   Decoder for the CTC
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

