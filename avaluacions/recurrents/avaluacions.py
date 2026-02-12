import torch
from jiwer import wer, cer

def evaluate_test_sample(sample, model, processor, device):
    if sample is None:
        raise ValueError("La mostra és buida o invàlida.")

    input_features, labels = sample
    input_features = input_features.unsqueeze(0).to(device)
    labels = labels.unsqueeze(0).to(device)

    prev_tokens = labels[:, :-1]
    target = labels[:, 1:]

    with torch.no_grad():
        output_logits = model(input_features, prev_tokens, teacher_forcing_ratio=0.0)
        predicted_ids = output_logits.argmax(dim=-1)

    pred_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    true_text = processor.batch_decode(target, skip_special_tokens=True)[0]

    wer_score = wer([true_text], [pred_text]) if true_text.strip() else None
    cer_score = cer([true_text], [pred_text]) if true_text.strip() else None

    return {
        "referència": true_text,
        "model": pred_text,
        "wer": wer_score,
        "cer": cer_score
    }