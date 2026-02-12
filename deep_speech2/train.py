import torch
import torch.nn as nn
import torch.nn.functional as F
from jiwer import wer,cer
from deep_speech2.tokenizer import GreedyDecoder


def train(model, device, dataloader, criterion, optimizer, scheduler, **kwags):
    model.train()
    data_len = len(dataloader.dataset)

    train_loss = 0
    for batch_idx, _data in enumerate(dataloader):
        spectrograms, labels, input_lengths, label_lengths = _data
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        # Check if any input length is shorter than target
        if any(i < t for i, t in zip(input_lengths, label_lengths)):
            continue

        optimizer.zero_grad()

        output = model(spectrograms)  # (batch, time, n_class)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1) # (time, batch, n_class)

        loss = criterion(output, labels, input_lengths, label_lengths)
        train_loss += loss.item()
        loss.backward()

        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if batch_idx % 100 == 0 or batch_idx == data_len:
            print(f"Batch {batch_idx+1}/{len(dataloader)} - Loss: {loss.item():.4f}")
            
    return {"train_loss": train_loss/len(dataloader)}


def test(model, device, dataloader, criterion, **kwargs):
    model.eval()
    test_loss = 0
    test_cer, test_wer = [], []
    with torch.no_grad():
        for I, _data in enumerate(dataloader):
            spectrograms, labels, input_lengths, label_lengths = _data
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1) # (time, batch, n_class)

            loss = criterion(output, labels, input_lengths, label_lengths)
            test_loss += loss.item() / len(dataloader)

            decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1), labels, label_lengths)
            for j in range(len(decoded_preds)):
                test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
                test_wer.append(wer(decoded_targets[j], decoded_preds[j]))


    avg_cer = sum(test_cer)/len(test_cer)
    avg_wer = sum(test_wer)/len(test_wer)

    return {"val_loss": test_loss, "val_wer": avg_wer, "val_cer": avg_cer}
