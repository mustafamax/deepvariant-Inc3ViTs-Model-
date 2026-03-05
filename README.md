# DeepVariant InceptionV3-LocalViT Keras Modeling

This repository contains a standalone `keras_modeling.py` file with a modified InceptionV3 backbone for DeepVariant.

## Architecture change

The main change is the addition of a **HybridViTBlock** after the InceptionV3 `mixed0` block:

- A `LocalPatchTokenizer` converts spatial feature maps into patch tokens.
- A `LocalSelfAttention` layer applies window-based self-attention over local regions.
- A lightweight MLP with residual connections refines the tokens.
- The output is reshaped back to `(B, H, W, C)` and concatenated with the original CNN features.

This design combines convolutional features with local Transformer-style attention while keeping the original Inception stem and DeepVariant training pipeline.

## Motivation

The goal of this modification is to:

- Reduce computation and latency by using local windowed self-attention instead of a heavy global Transformer.
- Preserve variant calling accuracy as much as possible by enriching InceptionV3 features with contextual information.

## How to use

To use this file inside DeepVariant:

1. Copy `deepvariant/keras_modeling.py` from this repository into your local DeepVariant source tree, replacing the original file.
2. In your training or inference config, set:
   ```python
   config.model_type = 'inception_v3'
