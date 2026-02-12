import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import torch
import torch.nn.functional as F
from models.ctc import DeepCTCModel
from models.ctc_millorat import DeepCTCModelV2
from ctc.dataloader import get_dataloaders
from avaluacions.ctc.avaluacions import evaluate_ctc_sample


def demo_evaluate_sample(index=[0]):
    dataset_path = "../../../../home/datasets/catalan_commonvoice/data"
    checkpoint_path = "checkpoints/ctc/CTC_tok_12.pth" 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carrega dataset i model
    _, _, test_set, tokenizer, num_classes = get_dataloaders(
        dataset_name=dataset_path,
        batch_size=1,
    )
    model = DeepCTCModel(num_classes=num_classes).to(device)
    #model = DeepCTCModelV2(num_classes=num_classes).to(device)
    model.eval()

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state_dict"])
    else:
        raise FileNotFoundError("Checkpoint no trobat.")

    # Avaluar mostres del test
    c = 0
    w = 0
    total = 0
    for i in range(len(test_set)):
        if i%100 == 0:
            print(f"{i}/{len(test_set)}")
        sample = test_set[i]
        result = evaluate_ctc_sample(sample, model, tokenizer, device)
        if result['wer'] and result['cer']:
            w += result['wer']
            c += result['cer']
            total += 1
    print(f"CER: {c/total}")
    print(f"WER: {w/total}")


if __name__ == "__main__":
    demo_evaluate_sample(index=list(range(0,10)))
