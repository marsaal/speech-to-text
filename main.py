import argparse
import json
from typing import Dict, Any

import ctc.train_ctc
import models.baseline
import models.attention
import models.ctc
import models.deep_speech
import models.ctc_millorat
import torch
import wandb
import os

import models
import recurrents.dataloader as rd
import recurrents.train as rt
import ctc.dataloader as ctcD
import ctc
import deep_speech2.dataloader as ds2d
import deep_speech2.train as ds2_t


model_functions = {
    "train": {
        "baseline": lambda p: rt.train(**p),
        "attention": lambda p: rt.train(**p),
        "CTC": lambda p: ctc.train_ctc.train(**p),
        "CTC-improved": lambda p: ctc.train_ctc.train(**p),
        "Deep-Speech-2": lambda p: ds2_t.train(**p),
    },
    "val": {
        "baseline": lambda p: rt.validate(**p),
        "attention": lambda p: rt.validate(**p),
        "CTC": lambda p: ctc.train_ctc.validate(**p),
        "CTC-improved": lambda p: ctc.train_ctc.validate(**p),
        "Deep-Speech-2": lambda p: ds2_t.test(**p),
    },
    "predict": {
        "baseline": lambda p: rt.train(**p),
        "attention": None,
        "CTC": None,
        "CTC-improved": None,
        "Deep-Speech-2": None,
    },
}

dataloaders = {
    "baseline": lambda p: rd.get_dataloaders(p["dataset_path"], p["batch_size"], p["frac"]),
    "attention": lambda p: rd.get_dataloaders(p["dataset_path"], p["batch_size"], p["frac"]),
    "CTC": lambda p: ctcD.get_dataloaders(p["dataset_path"], p["batch_size"], p["frac"]),
    "CTC-improved": lambda p: ctcD.get_dataloaders(p["dataset_path"], p["batch_size"], p["frac"]),
    "Deep-Speech-2": lambda p: ds2d.get_dataloaders(p["dataset_path"], p["batch_size"], p["frac"]),
}

models_dict = {
    "baseline": lambda num_classes: models.baseline.Baseline(num_classes=num_classes),
    "attention": lambda num_classes: models.attention.AttentionModel(num_classes=num_classes),
    "CTC": lambda num_classes: models.ctc.DeepCTCModel(num_classes=num_classes),
    "CTC-improved": lambda num_classes: models.ctc_millorat.DeepCTCModelV2(num_classes=num_classes),
    "Deep-Speech-2": lambda num_classes: models.deep_speech.SpeechRecognitionModel(num_classes=num_classes),
}

criterion_dict = {
    "baseline": lambda t: torch.nn.CrossEntropyLoss(ignore_index=0),
    "attention": lambda t: torch.nn.CrossEntropyLoss(ignore_index=0),
    "CTC": lambda t: torch.nn.CTCLoss(blank=t.convert_tokens_to_ids("<ctc_blank>"), zero_infinity=True),
    "CTC-improved": lambda t: torch.nn.CTCLoss(blank=t.convert_tokens_to_ids("<ctc_blank>"), zero_infinity=True),
    "Deep-Speech-2": lambda t: torch.nn.CTCLoss(blank=0),
}


def load_config(json_path: str) -> Dict[str, Any]:
    """Load and validate configuration from JSON file"""
    with open(json_path, 'r') as f:
        config = json.load(f)
    
    # Validate required fields
    required_fields = {
        'dataset_path', 'name', "batch_size", "learning_rate", "decay", "momentum",
        'checkpoint_path', 'checkpoint_folder', "n_epochs", "use_wandb",
        "resume_wandb", "frac"
    }
    
    missing_fields = required_fields - set(config.keys())
    if missing_fields:
        raise ValueError(f"Missing required fields in config: {missing_fields}")
    
    return config


def global_train(model_str, params: Dict[str, Any]):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(params["checkpoint_folder"], exist_ok=True)
    if params["use_wandb"]:
        if params["resume_wandb"] is None:
            wandb.init(project="asr-catalan____", config=params)
        else:
            wandb.init(
                project="asr-catalan____",
                config=params,
                id=params["resume_wandb"],
                resume="allow"          
            )

    train_loader, val_loader, _, processor, num_classes = dataloaders[model_str](params)

    model = models_dict[model_str](num_classes).to(device)

    scheduler = None
    if params["optimizer"].lower() == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=params["learning_rate"], weight_decay = params["decay"], momentum = params["momentum"])
    elif params["optimizer"].lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay = params["decay"])
    elif params["optimizer"].lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), params["learning_rate"])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
            max_lr=params["learning_rate"],
            steps_per_epoch=int(len(train_loader)),
            epochs=params["n_epochs"],
            anneal_strategy='linear'
        )
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=params["learning_rate"], weight_decay = params["decay"], momentum = params["momentum"])

    criterion = criterion_dict[model_str](processor)

    best_val_loss = float("inf")
    start_epoch = 0

    if params["resume"] and params["checkpoint_path"] and os.path.exists(params["checkpoint_path"]):
        checkpoint = torch.load(params["checkpoint_path"], map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint["val_loss"]
        print(f"Checkpoint carregat - Loss={best_val_loss:.4f}")


    for epoch in range(start_epoch, params["n_epochs"]):
        train_args = {
            "model": model,
            "dataloader": train_loader,
            "optimizer": optimizer,
            "criterion": criterion,
            "scheduler": scheduler,
            "device": device,
            "processor": processor,
            "tokenizer": processor,
            "teacher_forcing_ratio": params["teacher_forcing"]
        }
        val_args = {
            "model": model,
            "dataloader": val_loader,
            "criterion": criterion,
            "device": device,
            "processor": processor,
            "tokenizer": processor,
        }

        train_results = model_functions["train"][model_str](train_args)
        val_results = model_functions["val"][model_str](val_args)

        print(f"Epoch {epoch+1}/{params['n_epochs']}")
        print(train_results)
        print(val_results)

        if params["use_wandb"]:
            merged_dict = train_results | val_results
            wandb.log(merged_dict)

        torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_results["val_loss"]
        }, f"{params['checkpoint_folder']}{params['name']}_{epoch}.pth")
        print(f"Model millorat guardat")

    if params["use_wandb"]:
        wandb.finish()



def main():
    parser = argparse.ArgumentParser(
        description="XNAP project to convert Speech to Text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model Descriptions:
  baseline      A simple model using a basic encoder of a single CNN layer, a GRU encoder and a GRU decoder, and a classification layer that is a linear layer.
  attention     Similar architecture to the baeline model, with an addition of an Attention Layer before the decoder, to better analyze the context.
  CTC           Basic model based on the same architectures as the baseline model, using CTC technique for predictions
  CTC-improved  Similar as the CTC model, but deeper
  Deep-Speech-2 Model that mimics the architecture of the published model based on CTC named Deep Speech 2.
For more information, check the `architectures.md` file.
  
Examples:
  Train a baseline model:    python cli_model.py --model baseline --action train
  Make predictions with attention model: python cli_model.py -m attention -a predict
"""
    )

    # Add arguments
    parser.add_argument(
        '-m', '--model',
        required=True,
        choices=['baseline', 'attention', 'CTC', 'CTC-improved', "Deep-Speech-2"],
        help="Type of model to use"
    )
    parser.add_argument(
        '-a', '--action',
        required=True,
        choices=['train', 'predict', "test"],
        help="Action to perform with the model"
    )
    parser.add_argument(
        '-c', '--config',
        required=True,
        help="Path to JSON configuration file"
    )

    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


    if args.action == "train":
        global_train(args.model, config)


    

if __name__ == "__main__":
    main()
