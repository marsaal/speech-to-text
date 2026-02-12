import torch
import torch.nn.functional as F
from jiwer import wer, cer
import os
import random
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import time

# 🔤 Funció per decodificar prediccions CTC
def ctc_decode_batch(seqs, tokenizer):
    results = []
    for seq in seqs:
        text = ''.join([
            tokenizer.convert_ids_to_tokens(idx)
            for idx in seq
            if idx != tokenizer.pad_token_id
        ])
        results.append(text)
    return results


def train(model, dataloader, optimizer, scheduler, criterion, device, tokenizer,**kwargs):
    model.train()
    pred_texts = []
    true_texts = []
    total_loss = 0
    blank = tokenizer.convert_tokens_to_ids("<ctc_blank>")

    for batch_idx, batch in enumerate(dataloader):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        log_probs = F.log_softmax(logits, dim=2).transpose(0, 1)
        input_lengths = torch.full((logits.size(0),), logits.size(1), dtype=torch.long, device=device)
        target_lengths = torch.tensor([l[l != 0].size(0) for l in labels], dtype=torch.long, device=device)
        targets = torch.cat([l[l != 0] for l in labels])
        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
        pred_ids = log_probs.detach().argmax(dim=-1).transpose(0, 1)
        decoded_preds = []
        for seq in pred_ids:
            prev = -1
            new_seq = []
            for idx in seq.tolist():
                if idx != prev and idx != tokenizer.pad_token_id and idx!=blank:
                    new_seq.append(idx)
                prev = idx
            decoded_preds.append(new_seq)
        pred_texts.extend(ctc_decode_batch(decoded_preds, tokenizer))
        
        # ✅ Correcta decodificació de `labels` (sense ctc decoding)
        for label in labels:
            label_ids = [idx.item() for idx in label if idx.item() != tokenizer.pad_token_id]
            true_texts.append(''.join(tokenizer.convert_ids_to_tokens(label_ids)))
        
        if (batch_idx+1) % 50 == 0:
            print(f"Batch {batch_idx+1}/{len(dataloader)} - Loss: {loss.item():.4f}")

    train_wer = wer(true_texts, pred_texts)
    avg_loss = total_loss / batch_idx if batch_idx > 0 else float("inf")
    cer_score = cer(true_texts, pred_texts)

    return {"train_loss": avg_loss, "train_wer": train_wer, "train_cer": cer_score}

def validate(model, dataloader, criterion, device, tokenizer,**kwargs):
    model.eval()
    total_loss = 0.0
    pred_texts = []
    true_texts = []
    total_samples = 0
    blank = tokenizer.convert_tokens_to_ids("<ctc_blank>")
    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            logits = model(inputs)
            log_probs = F.log_softmax(logits, dim=2).transpose(0, 1)

            input_lengths = torch.full((logits.size(0),), logits.size(1), dtype=torch.long).to(device)
            target_lengths = torch.tensor([l[l != 0].size(0) for l in labels], dtype=torch.long).to(device)
            targets = torch.cat([l[l != 0] for l in labels])

            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            total_loss += loss.item()
            total_samples += 1

            pred_ids = log_probs.argmax(dim=-1).transpose(0, 1)
            decoded_preds = []
            for seq in pred_ids:
                prev = -1
                new_seq = []
                for idx in seq.tolist():
                    if idx != prev and idx != tokenizer.pad_token_id and idx!=blank:
                        new_seq.append(idx)
                    prev = idx
                decoded_preds.append(new_seq)

            pred_texts.extend(ctc_decode_batch(decoded_preds, tokenizer))
            # ✅ Correcta decodificació dels `labels`
            for label in labels:
                label_ids = [idx.item() for idx in label if idx.item() != tokenizer.pad_token_id]
                true_texts.append(''.join(tokenizer.convert_ids_to_tokens(label_ids)))

    val_wer = wer(true_texts, pred_texts)
    avg_loss = total_loss / total_samples if total_samples > 0 else float("inf")
    cer_score = cer(true_texts, pred_texts)

    return {"val_loss": avg_loss, "val_wer": val_wer, "val_cer": cer_score}
