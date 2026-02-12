# Speech-to-Text for Catalan

A PyTorch-based automatic speech recognition (ASR) system for Catalan language, implementing multiple neural network architectures including baseline models, attention mechanisms, CTC-based approaches, and Deep Speech 2.

## 🎯 Project Overview

This project explores different deep learning architectures for speech recognition on the Catalan Common Voice dataset. The implementation includes training, validation, and evaluation pipelines for five different model architectures, ranging from simple baseline models to state-of-the-art Deep Speech 2 inspired networks.

## 🏗️ Implemented Architectures

### 1. **Baseline Model**
A simple encoder-decoder model with:
- 1D CNN feature extraction
- Unidirectional GRU encoder
- GRU decoder with teacher forcing
- Sequential token generation

### 2. **Attention Model**
Enhanced baseline with:
- Improved 2-layer CNN feature extractor with BatchNorm and ReLU
- Attention mechanism to focus on relevant parts of the audio sequence
- Better context utilization during decoding

### 3. **DeepCTCModel**
CTC-based approach featuring:
- Convolutional feature extraction with batch normalization
- Bidirectional GRU encoder
- CTC loss for alignment-free training
- Linear classifier with dropout

### 4. **DeepCTCModelV2 (CTC-improved)**
Advanced CTC model with:
- Residual blocks for deep feature extraction
- 3-layer bidirectional GRU encoder
- 3-layer classifier network with progressive dimensionality reduction
- Enhanced gradient flow with residual connections

### 5. **Deep Speech 2**
State-of-the-art architecture inspired by [DeepSpeech2](https://github.com/sooftware/deepspeech2):
- 2D residual CNN blocks with LayerNorm
- Multi-layer bidirectional GRU stack
- GELU activations
- Robust classification head

For detailed architecture descriptions, see [arquitectures.md](arquitectures.md).

## 📊 Performance Metrics

The models are evaluated using:
- **WER (Word Error Rate)**: Measures word-level accuracy
- **CER (Character Error Rate)**: Measures character-level accuracy (particularly useful for CTC models)

Detailed results and analysis can be found in [resultats.md](resultats.md).

## 🚀 Getting Started

### Prerequisites

- Python 3.10.16
- CUDA-capable GPU (recommended)
- Common Voice Catalan dataset

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd spech2text
   ```

2. **Create conda environment**
   ```bash
   conda env create -f environment.yml
   conda activate xnap04
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

Edit `config.json` to customize training parameters:

```json
{
    "dataset_path": "path/to/catalan_commonvoice/data",
    "batch_size": 16,
    "frac": 1,
    "learning_rate": 1e-4,
    "decay": 1e-5,
    "momentum": 0.8,
    "optimizer": "adamw",
    "teacher_forcing": 0.5,
    "name": "s2t",
    "checkpoint_folder": "./checkpoints/",
    "resume": false,
    "checkpoint_path": "checkpoints/s2t_3.pth",
    "n_epochs": 10,
    "use_wandb": false,
    "resume_wandb": null
}
```

## 🎓 Training

### Train a Model

```bash
python main.py train --model [MODEL_NAME] --config config.json
```

Available models:
- `baseline`
- `attention`
- `CTC`
- `CTC-improved`
- `Deep-Speech-2`

### Example

```bash
python main.py train --model Deep-Speech-2 --config config.json
```

### Using Weights & Biases

To enable W&B logging, set `"use_wandb": true` in your config file:

```json
{
    "use_wandb": true,
    "resume_wandb": "run_id_or_null"
}
```

## 🧪 Evaluation

### Validate a Model

```bash
python main.py val --model [MODEL_NAME] --config config.json
```

### Run Evaluations

Each model type has its own evaluation scripts in the `avaluacions/` folder:

```bash
# CTC models
python avaluacions/ctc/avaluacions.py
python avaluacions/ctc/quantitatiu.py
python avaluacions/ctc/qualitatiu.py

# Deep Speech 2
python avaluacions/deep_speech2/avaluacions.py
python avaluacions/deep_speech2/per_accent.py

# Recurrent models (baseline/attention)
python avaluacions/recurrents/avaluacions.py
```

## 📁 Project Structure

```
spech2text/
├── main.py                  # Main entry point
├── config.json              # Configuration file
├── requirements.txt         # Python dependencies
├── environment.yml          # Conda environment
├── arquitectures.md         # Architecture documentation
├── resultats.md            # Results and analysis
├── models/                 # Model implementations
│   ├── baseline.py
│   ├── attention.py
│   ├── ctc.py
│   ├── ctc_millorat.py
│   └── deep_speech.py
├── processors/             # Audio feature extraction
│   └── feature_extraction.py
├── tokenizers/             # Tokenizer configurations
│   ├── ctc_catalan_char_tokenizer/
│   └── ctc_char/
├── ctc/                    # CTC model training
│   ├── dataloader.py
│   ├── generar_tokenizer.py
│   └── train_ctc.py
├── deep_speech2/           # Deep Speech 2 training
│   ├── dataloader.py
│   ├── tokenizer.py
│   └── train.py
├── recurrents/             # Baseline/Attention training
│   ├── dataloader.py
│   └── train.py
├── avaluacions/            # Evaluation scripts
│   ├── ctc/
│   ├── deep_speech2/
│   └── recurrents/
└── test/                   # Test scripts
```

## 🔧 Key Features

- **Multiple Architecture Support**: Train and compare different ASR architectures
- **CTC Loss Integration**: Alignment-free training for CTC models
- **Custom Tokenizers**: Character-level tokenizers optimized for Catalan
- **Feature Extraction**: Librosa-based audio preprocessing
- **Checkpoint Management**: Save and resume training
- **Comprehensive Evaluation**: Quantitative and qualitative analysis tools
- **W&B Integration**: Track experiments with Weights & Biases
- **GPU Support**: CUDA-accelerated training

## 📦 Dependencies

Key libraries:
- PyTorch 2.2.2 (with CUDA 12.1 support)
- Transformers (HuggingFace)
- Librosa (audio processing)
- Jiwer (WER/CER calculation)
- Datasets (data loading)
- Weights & Biases (experiment tracking)
- TensorBoard (visualization)

See [requirements.txt](requirements.txt) for the complete list.

## 📚 Dataset

This project uses the **Common Voice Catalan dataset**. Make sure to:
1. Download the dataset from [Common Voice](https://commonvoice.mozilla.org/)
2. Update the `dataset_path` in `config.json` to point to your dataset location

## 🧠 Model Selection Guide

- **Quick prototyping**: Start with `baseline`
- **Better accuracy**: Try `attention`
- **Alignment-free training**: Use `CTC` or `CTC-improved`
- **State-of-the-art performance**: Use `Deep-Speech-2`

## 📈 Monitoring Training

Training progress can be monitored through:
- **Console output**: Loss and metrics printed during training
- **TensorBoard**: Launch with `tensorboard --logdir=runs/`
- **Weights & Biases**: View experiments at [wandb.ai](https://wandb.ai)

## 🤝 Testing

Run project tests:

```bash
python test/test_basic.py
python test/check_required_files.py
python test/check_participation.py
```

## 📄 License

See [LICENSE](LICENSE) file for details.

## 🔍 Additional Documentation

- [arquitectures.md](arquitectures.md) - Detailed architecture descriptions
- [resultats.md](resultats.md) - Model performance and results
- [test.md](test.md) - Testing information

## 💡 Tips

1. **GPU Memory**: Reduce `batch_size` if you encounter OOM errors
2. **Teacher Forcing**: Experiment with different schedules for attention models
3. **Learning Rate**: Use lower learning rates for fine-tuning
4. **Dataset Size**: Adjust `frac` parameter to use a subset of data for quick experiments
5. **Checkpointing**: Regularly save checkpoints for long training runs

## 🐛 Troubleshooting

**CUDA out of memory**:
- Reduce batch size in `config.json`
- Use gradient accumulation

**Poor performance**:
- Check dataset path is correct
- Verify tokenizer configuration
- Increase number of epochs
- Experiment with learning rate

**Import errors**:
- Ensure conda environment is activated
- Reinstall requirements: `pip install -r requirements.txt`

---

*Project developed for XNAP course - Speech Recognition for Catalan*
