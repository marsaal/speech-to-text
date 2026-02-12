import torch
from jiwer import wer, cer
import wandb
import numpy as np


def train(model, dataloader, optimizer, scheduler, criterion, device, processor,teacher_forcing_ratio=0, **kwargs):
    model.train()
    total_loss = 0.0
    pred_texts = []
    true_texts = []
    total_samples = 0  # comptador de mostres útils

    for batch_idx, batch in enumerate(dataloader):
        if batch is None:
            continue
        inputs, labels = batch
        batch_size = inputs.size(0)
        total_samples += batch_size

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        prev_tokens = labels[:, :-1]
        target = labels[:, 1:]
        outputs = model(inputs, prev_tokens,teacher_forcing_ratio)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        loss = criterion(outputs.view(-1, outputs.shape[-1]), target.reshape(-1))
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()*batch_size

        if (batch_idx + 1) % 250 == 0:
            print(f"Batch {batch_idx+1}/{len(dataloader)} - Loss: {loss.item():.4f}")
        
        
        # Decodificació per WER
        predicted_ids = outputs.argmax(dim=-1)
        pred_texts.extend(processor.batch_decode(predicted_ids, skip_special_tokens=True))
        true_texts.extend(processor.batch_decode(target, skip_special_tokens=True))
    # Filter out entries where the reference is an empty string
    filtered_true_texts = []
    filtered_pred_texts = []
    for ref, hyp in zip(true_texts, pred_texts):
        if ref.strip():  # keep if reference is not empty
            filtered_true_texts.append(ref)
            filtered_pred_texts.append(hyp)

    train_wer = wer(filtered_true_texts, filtered_pred_texts)
    train_cer = cer(filtered_true_texts, filtered_pred_texts)

    avg_loss = total_loss / total_samples if total_samples > 0 else float("inf")

    return {"train_loss": avg_loss, "train_wer": train_wer, "train_cer": train_cer}



def validate(model, dataloader, criterion, device, processor, **kwargs):
    model.eval()
    total_loss = 0.0
    pred_texts = []
    true_texts = []
    total_samples = 0  # Comptador de mostres útils

    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
            inputs, labels = batch
            batch_size = inputs.size(0)
            total_samples += batch_size

            inputs, labels = inputs.to(device), labels.to(device)

            prev_tokens = labels[:, :-1]
            target = labels[:, 1:]

            outputs = model(inputs, prev_tokens)
            loss = criterion(outputs.view(-1, outputs.shape[-1]), target.reshape(-1))
            total_loss += loss.item()*batch_size

            predicted_ids = outputs.argmax(dim=-1)
            pred_texts.extend(processor.batch_decode(predicted_ids, skip_special_tokens=True))
            true_texts.extend(processor.batch_decode(target, skip_special_tokens=True))
    
    # Filtrar entrades amb referència buida
    filtered_true_texts = []
    filtered_pred_texts = []
    for ref, hyp in zip(true_texts, pred_texts):
        if ref.strip():  # mantenir si la referència no és buida
            filtered_true_texts.append(ref)
            filtered_pred_texts.append(hyp)

    avg_loss = total_loss / total_samples if total_samples > 0 else float("inf")
    wer_score = wer(filtered_true_texts, filtered_pred_texts)
    cer_score = cer(filtered_true_texts, filtered_pred_texts)

    return {"val_loss": avg_loss, "val_wer": wer_score, "val_cer": cer_score}

