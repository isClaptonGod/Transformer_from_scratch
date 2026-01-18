
# Transformer From Scratch (PyTorch)

An implementation of "Attention Is All You Need" (Vaswani et al., 2017) focusing on readability and reproducibility for Machine Translation.

## Key Features
- **Multi-Head Attention**: Direct implementation of Scaled Dot-Product Attention.
- **Positional Encoding**: Sinusoid-based position awareness.
- **Masking**: Look-ahead masking for decoder and padding masking for encoder.
- **Dataset**: Multi30k (German to English).

## Results
| Model | BLEU Score |
| :--- | :--- |
| This Implementation | 32.5 (Example) |
| Paper (Base) | 27.3 |

## How to Run
1. `pip install -r requirements.txt`
2. `python train.py`
