import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import torch
from models.baseline import Baseline
from models.attention import AttentionModel
from recurrents.dataloader import get_dataloaders
from avaluacions.recurrents.avaluacions import evaluate_test_sample

def demo_evaluate_sample():
    dataset_path = "../../../../home/datasets/catalan_commonvoice/data"
    _, _, test_set, processor, num_classes = get_dataloaders(dataset_name=dataset_path, batch_size=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "checkpoints/latest/latestbaseline_10.pth" # Baseline 
    #checkpoint_path = "checkpoints/attention/model2_13.pth" # Attention 
    print(f"Token count: {len(processor.tokenizer)}")

    
    input_dim, hidden_dim, embedding_dim  = 80, 256, 64

    # Carrega dataset i model
    model = Baseline(input_dim=input_dim, hidden_dim=hidden_dim, embedding_dim=embedding_dim, num_classes=num_classes).to(device)
    #model = AttentionModel(input_dim, hidden_dim, embedding_dim, num_classes).to(device)
    model.eval()
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state_dict"])
    else:
        raise FileNotFoundError("Checkpoint no trobat.")
    c = 0
    w = 0
    total = 0
    # Avaluar una mostra del test
    for i in range(len(test_set)):
        print(f"{i}/{len(test_set)}")
        sample = test_set[i]
        result = evaluate_test_sample(sample, model, processor, device)
        if result['wer'] and result['cer']:
            w += result['wer']
            c += result['cer']
            total += 1
    print(f"CER: {c/total}")
    print(f"WER: {w/total}")

if __name__ == "__main__":
    demo_evaluate_sample()