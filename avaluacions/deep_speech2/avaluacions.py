import torch
import torch.nn as nn
import torch.nn.functional as F
from jiwer import wer,cer
from deep_speech2.tokenizer import GreedyDecoder
import os
import numpy as np
from deep_speech2.dataloader import get_dataloaders, get_test_dataloaders_per_accent
from models.deep_speech import SpeechRecognitionModel
from deep_speech2.tokenizer import TextTransform


def test_single(sample, model, device):
    with torch.no_grad():
        spectrograms, labels, _, label_lengths = sample
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        output = model(spectrograms)  # (batch, time, n_class)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1) # (time, batch, n_class)

        decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1), labels, label_lengths)

        return decoded_preds[0], wer(decoded_targets[0], decoded_preds[0]), cer(decoded_targets[0], decoded_preds[0])


def evaluate_test_sample(sample, model, device, tokenizer):
    if sample is None:
        raise ValueError("La mostra és buida o invàlida.")

    pred_text, wers, cers = test_single(sample, model, device)

    return {
        "referència": tokenizer.int_to_text(sample[1].unsqueeze(0).squeeze().tolist()),
        "model": pred_text,
        "wer": wers,
        "cer": cers
    }

def demo_evaluate_sample(n_classes, index=None, test_loaders_map={}, min_cer_to_print=0.0, idx_to_print=[0]):
    tokenizer = TextTransform()
    checkpoint_path = "deep_speech2/checkpoints/CTC_TOP_8.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carrega dataset i model
    model = SpeechRecognitionModel(
       n_classes
    )
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])
    else:
        raise FileNotFoundError("Checkpoint no trobat.")
    
    model = model.to(device)
    model.eval()

    for (accent, dataloader) in test_loaders_map.items():
        print(f"Evaluating accent: {accent}")
        wer, cer = [], []
        # Avaluar una mostra del test
        for batch_idx, data in enumerate(dataloader):
            if index is not None and batch_idx not in index:
                break

            try:
                result = evaluate_test_sample(data, model, device, tokenizer)
                wer.append(result['wer'])
                cer.append(result['cer'])

                if result["cer"] > min_cer_to_print and batch_idx in idx_to_print:
                    print(f"\nExemple {batch_idx}")
                    print(f"Referència  : {result['referència']}")
                    print(f"Model       : {result['model']}")
                    print(f"WER         : {result['wer']:.4f}")
                    print(f"CER         : {result['cer']:.4f}")
            except:
                print("Error with sample ", batch_idx)

        print(f"Mean WER: {sum(wer)/len(wer):.4f}. Std WER: {np.std(wer)}")
        print(f"Mean CER: {sum(cer)/len(cer):.4f}. Std CER: {np.std(cer)}")

if __name__ == "__main__":
    dataset_path = "../../../../home/datasets/catalan_commonvoice/data"

    _, _, test_loader, _, n_classes = get_dataloaders(dataset_name=dataset_path, batch_size=1, frac=1)
    all_accents = {"all_accents": test_loader}

    accents_separated = get_test_dataloaders_per_accent(dataset_path)

    demo_evaluate_sample(n_classes, None, all_accents, min_cer_to_print=0, idx_to_print=list(range(10)))


