from torch.utils.data import Dataset, DataLoader 
from datasets import load_dataset, Audio
from transformers import WhisperFeatureExtractor, PreTrainedTokenizerFast
import torch
from processors.feature_extraction import normalize_audio, trim_silence, prepare_inputs, tokenize_transcript, load_audio
import torchaudio
import random

class AudioDataset(Dataset):
    def __init__(self, dataset, feature_extractor, tokenizer, sampling_rate=16000):
        self.dataset = dataset
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            example = self.dataset[idx]
            waveform = torch.tensor(example["audio"]["array"], dtype=torch.float32)
            sr = example["audio"]["sampling_rate"]
            transcript = example["sentence"]
            if sr != self.sampling_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sampling_rate)
                waveform = resampler(waveform)

            waveform = normalize_audio(waveform)
            waveform = trim_silence(waveform, self.sampling_rate)

            inputs = prepare_inputs(waveform, self.feature_extractor, sampling_rate=self.sampling_rate)
            input_features = inputs.input_features.squeeze(0)

            labels = tokenize_transcript(transcript, self.tokenizer)
            if labels.dim() > 1:
                labels = labels.squeeze(0)

            return input_features, labels

        except Exception as e:
            print(f"[WARN] Error en mostra {idx}: {e}")
            return None

def tokenize_transcript(text, tokenizer):
    encoding = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    return encoding.input_ids[0]  # tensor 1D compatible amb CTC

def prepare_inputs(waveform, feature_extractor, sampling_rate=16000):
    return feature_extractor(waveform, sampling_rate=sampling_rate, return_tensors="pt")

# ✅ collate_fn tancada sobre tokenizer per padding correcte
def build_collate_fn(tokenizer):
    def collate_fn(batch):
        input_features, labels = zip(*batch)
        input_features = torch.nn.utils.rnn.pad_sequence(input_features, batch_first=True)
        pad_id = tokenizer.pad_token_id
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=pad_id)

        return input_features, labels
    return collate_fn

def get_dataloaders(dataset_name, batch_size=32, frac=1, seed=42):
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    tokenizer = PreTrainedTokenizerFast.from_pretrained("tokenizers/ctc_catalan_char_tokenizer")
    num_classes = tokenizer.vocab_size

    dataset = load_dataset(
        "parquet",
        data_files={
            "train": f"{dataset_name}/train/*",
            "validation": f"{dataset_name}/validation-*.parquet",
            "test": f"{dataset_name}/test/*"
        },
        cache_dir=dataset_name
    )
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    random.seed(seed)
    dataset_train = dataset["train"].shuffle(seed=seed).select(range(int(len(dataset["train"]) * frac)))
    dataset_val = dataset["validation"].shuffle(seed=seed).select(range(int(len(dataset["validation"]) * frac)))
    dataset_test = dataset["test"].shuffle(seed=seed).select(range(int(len(dataset["test"]) * frac)))

    train_set = AudioDataset(dataset_train, feature_extractor, tokenizer)
    val_set = AudioDataset(dataset_val, feature_extractor, tokenizer)
    test_set = AudioDataset(dataset_test, feature_extractor, tokenizer)

    collate = build_collate_fn(tokenizer)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=10)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=10)

    return train_loader, val_loader, test_set, tokenizer, num_classes
