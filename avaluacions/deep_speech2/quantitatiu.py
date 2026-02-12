import avaluacions.deep_speech2.avaluacions as a
from deep_speech2.dataloader import get_dataloaders

if __name__ == "__main__":
    dataset_path = "../../../../home/datasets/catalan_commonvoice/data"

    _, _, test_loader, _, n_classes = get_dataloaders(dataset_name=dataset_path, batch_size=1, frac=1)
    all_accents = {"all_accents": test_loader}

    a.demo_evaluate_sample(n_classes, None, all_accents, min_cer_to_print=0, idx_to_print=[])
    