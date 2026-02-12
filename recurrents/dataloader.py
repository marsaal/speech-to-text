# --- dataloader.py ---
from torch.utils.data import Dataset, DataLoader 
from datasets import load_dataset, Audio
from transformers import Speech2TextProcessor
import torch
from processors.feature_extraction import normalize_audio, trim_silence, prepare_inputs, tokenize_transcript, load_audio
import torchaudio
import random

class AudioDataset(Dataset):
    def __init__(self, dataset, processor, sampling_rate=16000):
        self.dataset = dataset
        self.processor = processor
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

            if waveform.numel() < 400:
                return None  

            inputs = prepare_inputs(waveform, self.processor, sampling_rate=self.sampling_rate)
            input_features = inputs.input_features.squeeze(0)

            labels = tokenize_transcript(transcript, self.processor)
            if labels.dim() > 1:
                labels = labels.squeeze(0)

            return input_features, labels

        except Exception as e:
            print(f"[WARN] Error en mostra {idx}: {e}")
            return None

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None

    input_features, labels = zip(*batch)
    input_features = torch.nn.utils.rnn.pad_sequence(input_features, batch_first=True)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)

    return input_features, labels

def get_dataloaders(dataset_name, batch_size=32, frac=1, seed=42):
    processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")
    num_classes = len(processor.tokenizer)

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

    train_set = AudioDataset(dataset_train, processor)
    val_set = AudioDataset(dataset_val, processor)
    test_set = AudioDataset(dataset_test, processor)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=10)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=10)

    return train_loader, val_loader, test_set, processor, num_classes

def get_test_dataloaders_per_accent(dataset_name, frac=1, seed=42):
    processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")
    num_classes = len(processor.tokenizer)
    accents = ["central", "nord-occidental", "valencià", "balear", "septentrional"]
    
    dataset = load_dataset(
        "parquet",
        data_files={
            "test": f"{dataset_name}/test/*"
        },
        cache_dir=dataset_name
    )
    
    random.seed(seed)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset_test = dataset["test"].shuffle(seed=seed).select(range(int(len(dataset["test"]) * frac)))
    
    maps = {}
    for accent in accents:
        accent_test_dataset = dataset_test.filter(lambda x: x["accent"] == accent, num_proc=8)
        test_set = AudioDataset(accent_test_dataset, processor)
        maps[accent] = test_set
    return maps, processor, num_classes

