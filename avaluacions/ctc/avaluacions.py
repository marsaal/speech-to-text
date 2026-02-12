import torch
import torch.nn.functional as F
from jiwer import wer,cer

def evaluate_ctc_sample(sample, model, tokenizer, device):
    if sample is None:
        raise ValueError("La mostra és buida o invàlida.")

    input_features, labels = sample
    input_features = input_features.unsqueeze(0).to(device)
    labels = labels.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_features)
        log_probs = F.log_softmax(logits, dim=2)
        pred_ids = log_probs.argmax(dim=-1)[0]  # (time,)

    # ⚡ Decodificació CTC optimitzada
    blank_id = tokenizer.convert_tokens_to_ids("<ctc_blank>")
    pad_id = tokenizer.pad_token_id

    decoded_seq = []
    prev = -1
    for idx in pred_ids:
        idx = idx.item()
        if idx != prev and idx != pad_id and idx != blank_id:
            decoded_seq.append(idx)
        prev = idx

    # ✅ Una sola conversió per seqüència
    pred_tokens = tokenizer.convert_ids_to_tokens(decoded_seq)
    pred_text = ''.join(pred_tokens)

    # 🔁 Etiquetes reals
    true_ids = [idx for idx in labels[0].tolist() if idx != pad_id]
    true_text = ''.join(tokenizer.convert_ids_to_tokens(true_ids))

    wers = wer([true_text], [pred_text]) if true_text.strip() else None
    cers = cer([true_text], [pred_text]) if true_text.strip() else None

    return {
        "referència": true_text,
        "model": pred_text,
        "wer": wers,
        "cer": cers
    }