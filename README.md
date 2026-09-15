# Tiny GPT Attention Experiments

This repository contains a from-scratch character-level Transformer/GPT experiment on the **Tiny Shakespeare** dataset, comparing three attention mechanisms:

1. **Dense causal attention**
2. **Sliding-window causal attention**
3. **BigBird-style sparse attention**

## Files

```text
Attention.ipynb
Tiny_GPT_Ai1.ipynb
Tiny_GPT_Ai2.ipynb
Tiny_GPT_Ai3.ipynb
input.txt
README.md
```

`Attention.ipynb` contains the three attention implementations, correctness checking, and attention benchmarking to be run on CPU.

The three `Tiny_GPT_Ai*.ipynb` notebooks contain the Tiny Shakespeare character-level GPT training experiments. The number in the filename identifies the attention variant used by that experiment.

## Requirements

The notebooks are written for **Python + PyTorch**, and can be run in Google Colab.

A GPU is strongly recommended for GPT training. In Colab:

```text
Runtime → Change runtime type → T4 GPU
```

The code automatically selects CUDA when it is available.

## Dataset

The GPT notebooks expect a file named:

```text
input.txt
```

containing the Tiny Shakespeare text.

The notebooks perform character-level tokenization:

```text
character → integer ID
```

The dataset is converted into a PyTorch tensor and the vocabulary is constructed from the unique characters.

## 1. Run the attention experiments

Open:

```text
Attention.ipynb
```

Run the notebook from the beginning.

It contains:

- `Attention_1` — dense causal attention
- `Attention_2` — sliding-window causal attention
- `Attention_3` — BigBird-like attention
- correctness comparison
- runtime and memory benchmarking

The benchmarking section compares the three attention mechanisms over increasing sequence lengths.

## 2. Run the GPT experiments

Run the three GPT notebooks separately:

```text
Tiny_GPT_Ai1.ipynb
Tiny_GPT_Ai2.ipynb
Tiny_GPT_Ai3.ipynb
```

Run each notebook **from top to bottom**.

Each notebook:

1. Loads `input.txt`
2. Builds the character vocabulary
3. Converts the text to integer IDs
4. Creates token and positional embeddings
5. Builds a 2-layer Transformer
6. Projects the hidden states into Q, K and V
7. Applies the notebook's attention mechanism in both Transformer layers
8. Applies feed-forward networks and residual connections
9. Produces vocabulary logits with the language-model head
10. Uses next-character targets and cross-entropy loss
11. Performs backpropagation with Adam
12. Trains on the GPU when CUDA is available