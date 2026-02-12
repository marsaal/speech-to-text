import avaluacions.deep_speech2.avaluacions as a
from deep_speech2.dataloader import get_dataloaders, get_test_dataloaders_per_accent

if __name__ == "__main__":
    dataset_path = "../../../../home/datasets/catalan_commonvoice/data"

    _, _, test_loader, _, n_classes = get_dataloaders(dataset_name=dataset_path, batch_size=1, frac=1)
    accents_separated = get_test_dataloaders_per_accent(dataset_path)

    a.demo_evaluate_sample(n_classes, None, accents_separated, min_cer_to_print=0, idx_to_print=[])
    