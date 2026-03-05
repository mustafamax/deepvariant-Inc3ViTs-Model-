# DeepVariant InceptionV3-LocalViT Keras Modeling

This repository contains a standalone `deepvariant/keras_modeling.py` file with a modified InceptionV3 backbone for DeepVariant.

The main change is the addition of a **HybridViTBlock** (local windowed self-attention + MLP) after the `mixed0` Inception block. The goal is to speed up the model while preserving variant calling accuracy as much as possible, by combining CNN features with lightweight local Transformer-style attention.

The file is designed to plug into the original DeepVariant codebase in place of the default `keras_modeling.py`.
