import sys
import os
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import torch
from models.baseline import Baseline
from models.attention import AttentionModel
from recurrents.dataloader import get_dataloaders
from avaluacions.recurrents.avaluacions import evaluate_test_sample

def demo_evaluate_sample(index=[0]):
    dataset_path = "../../../../home/datasets/catalan_commonvoice/data"
    checkpoint_path = "checkpoints/latest/latestbaseline_10.pth" # Baseline 
    #checkpoint_path = "checkpoints/attention/model2_13.pth" # Attention 
    input_dim, hidden_dim, embedding_dim = 80, 256, 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carrega dataset i model
    _, _, test_set, processor, num_classes = get_dataloaders(dataset_name=dataset_path, batch_size=1)
    
    model = Baseline(input_dim=input_dim, hidden_dim=hidden_dim, embedding_dim=embedding_dim, num_classes=num_classes).to(device)
    #model = AttentionModel(input_dim, hidden_dim, embedding_dim, num_classes).to(device)
    model.eval()
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state_dict"])
    else:
        raise FileNotFoundError("Checkpoint no trobat.")

    # Avaluar una mostra del test
    for i in index:
        sample = test_set[i]
        result = evaluate_test_sample(sample, model, processor, device)

        print(f"\nExemple {i}")
        print(f"Referència  : {result['referència']}")
        print(f"Model       : {result['model']}")
        print(f"WER         : {result['wer']:.4f}")
        print(f"CER         : {result['cer']:.4f}")

if __name__ == "__main__":
    demo_evaluate_sample(index=[i for i in range(0,10)])