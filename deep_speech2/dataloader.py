from torch.utils.data import Dataset, DataLoader 
from datasets import load_dataset, Audio
from transformers import WhisperFeatureExtractor, PreTrainedTokenizerFast
import torch
from processors.feature_extraction import normalize_audio, trim_silence, prepare_inputs, tokenize_transcript, load_audio
import torchaudio
import random
from deep_speech2.tokenizer import vocab_list, data_processing
import io

class AudioDataset(Dataset):
    def __init__(self, dataset, sampling_rate=16000):
        self.dataset = dataset
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            audio = self.dataset[idx]["audio"]
            transcription = self.dataset[idx]["sentence"]
            
            # Decode from raw bytes
            audio_bytes = audio['bytes']
            audio_stream = io.BytesIO(audio_bytes)
            waveform, sample_rate = torchaudio.load(audio_stream)
            
            if sample_rate != self.sampling_rate:
                waveform = torchaudio.functional.resample(waveform, sample_rate, self.sampling_rate)

            return waveform, transcription

        except Exception as e:
            print(f"[WARN] Error en mostra {idx}: {e}")
            return None

def get_dataloaders(dataset_name, batch_size=32, frac=1, seed=42):
    num_classes = len(vocab_list) + 2

    dataset = load_dataset(
        "parquet",
        data_files={
            "train": f"{dataset_name}/train/*",
            "validation": f"{dataset_name}/validation-*.parquet",
            "test": f"{dataset_name}/test/*"
        },
        cache_dir=dataset_name
    )

    random.seed(seed)
    dataset_train = dataset["train"].shuffle(seed=seed).select(range(int(len(dataset["train"]) * frac)))
    dataset_val = dataset["validation"].shuffle(seed=seed).select(range(int(len(dataset["validation"]) * frac)))
    dataset_test = dataset["test"].shuffle(seed=seed).select(range(int(len(dataset["test"]) * frac)))

    train_set = AudioDataset(dataset_train)
    val_set = AudioDataset(dataset_val)
    test_set = AudioDataset(dataset_test)

    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=lambda x: data_processing(x, "train"), 
        num_workers=10
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=lambda x: data_processing(x, "valid"), 
        num_workers=10
    )
    test_loader = DataLoader(
        test_set, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=lambda x: data_processing(x, "test"), 
        num_workers=10
    )

    return train_loader, val_loader, test_loader, None, num_classes


def get_test_dataloaders_per_accent(dataset_name, frac=1, seed=42):
    num_classes = len(vocab_list) + 2
    accents = ["central", "nord-occidental", "valencià", "balear", "septentrional"]
    
    dataset = load_dataset(
        "parquet",
        data_files={
            "test": f"{dataset_name}/test/*"
        },
        cache_dir=dataset_name
    )

    random.seed(seed)
    dataset_test = dataset["test"].shuffle(seed=seed).select(range(int(len(dataset["test"]) * frac)))
    
    maps = {}
    for accent in accents:
        accent_test_dataset = dataset_test.filter(lambda x: x["accent"] == accent, num_proc=8)
        test_set = AudioDataset(accent_test_dataset)
        test_loader = DataLoader(
            test_set, 
            batch_size=1, 
            shuffle=False, 
            collate_fn=lambda x: data_processing(x, "test"), 
            num_workers=10
        )
        maps[accent] = test_loader
    return maps