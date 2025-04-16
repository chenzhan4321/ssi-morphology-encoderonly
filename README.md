# Syriac Morphology Analysis System

This project implements a deep learning-based system for analyzing Syriac text morphology using a transformer-based neural network architecture. The system is designed to process Syriac text and predict morphological patterns, providing detailed annotations for each character in the input text.

## Project Overview

The system consists of several key components:

1. **Data Processing**: Handles Syriac text input and converts it into a format suitable for neural network processing
2. **Neural Network Model**: A transformer-based architecture with character-level CNN features
3. **Training Pipeline**: Implements advanced training techniques including mixed precision training and class balancing
4. **Inference System**: Processes new text and generates morphological annotations

## Key Features

- **Advanced Architecture**: Combines transformer encoders with character-level CNNs for robust feature extraction
- **Positional Encoding**: Implements sinusoidal positional encoding for sequence position awareness
- **Class Balancing**: Special handling for imbalanced class distributions
- **Mixed Precision Training**: Utilizes FP16 training for improved performance
- **Intelligent Text Splitting**: Smart sentence segmentation while preserving context
- **Comprehensive Metrics**: Tracks multiple accuracy metrics including zero/non-zero classification

## Requirements

- Python 3.x
- PyTorch
- pandas
- numpy
- tqdm
- Levenshtein

Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `model.py`: Contains the neural network architecture implementation
- `train.py`: Training pipeline and model evaluation
- `dataset.py`: Data loading and preprocessing
- `parse.py`: Inference and text processing utilities
- `patterns.csv`: Contains mapping between numeric labels and morphological patterns
- `training.csv`: Training data in CSV format

## Usage

### Training

To train the model:
```bash
python train.py
```

The training script supports various parameters:
- Learning rate scheduling
- Mixed precision training
- Gradient clipping
- Multiple accuracy metrics tracking

Advanced training options:
```bash
python train.py --batch_size 32 --num_epochs 100 --learning_rate 0.0001
```

Training options:
- `--batch_size`: Batch size for training (default: 32)
- `--num_epochs`: Number of training epochs (default: 50)
- `--learning_rate`: Initial learning rate (default: 0.0001)
- `--model_path`: Path to save the trained model (default: 'best_model.pth')
- `--data_path`: Path to training data (default: 'training.csv')

### Inference

To process new text:
```bash
python parse.py --input "Your Syriac text here"
```

Or process a file:
```bash
python parse.py --file input.txt --output output.txt
```

Advanced inference options:
```bash
python parse.py --model path/to/model.pth --file input.txt --output custom_output.txt
```

## Model Architecture

The model combines several advanced techniques:

1. **Character Embedding**: Maps Syriac characters to dense vectors
2. **Positional Encoding**: Adds positional information to embeddings
3. **Character-level CNN**: Extracts local features from character sequences
4. **Transformer Encoder**: Processes sequence-level features
5. **Classification Head**: Predicts morphological patterns

## Performance Metrics

The system tracks multiple accuracy metrics:
- Zero to Zero accuracy
- Non-zero to Non-zero accuracy
- Non-zero exact match accuracy
- Overall accuracy
- Loss metrics

## Data Format

### Input Format
The system expects input text in Syriac characters, with the following mapping:
```
'>': 0, 'B': 1, 'G': 2, 'D': 3, 'H': 4, 'W': 5, 'Z': 6,
'X': 7, 'V': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13,
'S': 14, '<': 15, 'P': 16, 'Y': 17, 'Q': 18, 'R': 19,
'C': 20, 'T': 21, ' ': 22
```

Syriac characters and their corresponding Latin transcriptions:
```
ܐ (Aleph) -> '>'
ܒ (Beth) -> 'B'
ܓ (Gamal) -> 'G'
ܕ (Dalath) -> 'D'
ܗ (He) -> 'H'
ܘ (Waw) -> 'W'
ܙ (Zain) -> 'Z'
ܚ (Heth) -> 'X'
ܛ (Teth) -> 'V'
ܝ (Yodh) -> 'J'
ܟ (Kaph) -> 'K'
ܠ (Lamadh) -> 'L'
ܡ (Mem) -> 'M'
ܢ (Nun) -> 'N'
ܣ (Semkath) -> 'S'
ܥ (E) -> '<'
ܦ (Pe) -> 'P'
ܨ (Sadhe) -> 'Y'
ܩ (Qoph) -> 'Q'
ܪ (Resh) -> 'R'
ܫ (Shin) -> 'C'
ܬ (Taw) -> 'T'
```

### Output Format
The system generates morphological annotations for each character in the input text, with patterns defined in `patterns.csv`.

## SOTA Benchmarks:
At default, the model reaches on c.a. 128th epoch at the best result:

- Zero/Non-zero Ratio: 0.8372/0.1628
- Zero to Zero Accuracy: 0.9760
- Non-zero to Non-zero Accuracy: 0.9776
- Non-zero Exact Match Accuracy: 0.8816
- Overall Accuracy: 0.9607
- Average Levenshtein Distance: 0.0452
