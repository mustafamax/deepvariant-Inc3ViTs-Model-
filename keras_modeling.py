import os
from typing import Callable, Optional, Sequence, Tuple, Type, Union

from absl import logging
import ml_collections
import numpy as np
import tensorflow as tf

from deepvariant import dv_constants
from deepvariant import metrics

_DEFAULT_WEIGHT_DECAY = 0.00004
_DEFAULT_BACKBONE_DROPOUT_RATE = 0.2


def build_classification_head(
    inputs: tf.Tensor,
    l2: float = 0.0,
) -> tf.Tensor:
  """Builds the classification head on top of a backbone feature tensor.

  This head is currently configured for multi-class classification and can be
  extended in the future for regression or multi-head setups.

  Args:
    inputs: Backbone output tensor used as input to the classification head.
    l2: L2 regularization factor for Dense layers.

  Returns:
    Output tensor representing class probabilities.
  """
  l2_regularizer = tf.keras.regularizers.L2(l2) if l2 else None
  head = tf.keras.layers.Dense(
      dv_constants.NUM_CLASSES,
      activation='softmax',
      dtype=tf.float32,
      name='classification',
      kernel_regularizer=l2_regularizer,
  )
  return head(inputs)


def add_l2_regularizers(
    model: tf.keras.Model,
    layer_class: Type[tf.keras.layers.Layer],
    l2: float = _DEFAULT_WEIGHT_DECAY,
) -> tf.keras.Model:
  """Adds L2 regularizers to all `layer_class` layers in `model`.

  Models from `tf.keras.applications` do not support specifying kernel or bias
  regularizers directly. To enable weight decay when fine-tuning ImageNet
  pretrained backbones, this helper attaches L2 losses to existing layers.

  Note:
    Existing `kernel_regularizer` attributes are not overwritten.

  Args:
    model: Base Keras model.
    layer_class: Layer type to which L2 regularization is applied.
    l2: L2 regularization factor.

  Returns:
    Model with additional L2 losses registered on the selected layers.
  """
  if not l2:
    return model

  num_regularizers_added = 0

  def add_l2_regularization(layer):
    def _add_l2():
      l2_reg = tf.keras.regularizers.l2(l2=l2)
      return l2_reg(layer.kernel)
    return _add_l2

  for layer in model.layers:
    if isinstance(layer, layer_class):
      model.add_loss(add_l2_regularization(layer))
      num_regularizers_added += 1

  logging.info('Added %d regularizers.', num_regularizers_added)
  return model


def load_weights_to_model_with_different_channels(
    model: tf.keras.Model,
    input_model: tf.keras.Model,
) -> tf.keras.Model:
  """Initializes `model` from `input_model` when channel counts differ.

  This utility copies weights layer-by-layer from an input model to a target
  model that shares the same architecture, but may differ in the number of
  input channels. When channel dimensions do not match, the minimum number of
  channels is used.

  Args:
    model: Target model to be initialized.
    input_model: Source model providing weights.

  Returns:
    Model with updated weights.
  """
  for layer_i, (input_model_layer, new_layer) in enumerate(
      zip(input_model.layers, model.layers)
  ):
    if not new_layer.weights:
      continue
    if len(new_layer.weights) != len(input_model_layer.weights):
      raise ValueError(
          'We expect both models to share the same InceptionV3 topology. '
          'Layer weight structures do not match.'
      )

    new_weights_to_assign = new_layer.get_weights()

    for i, (input_weights, target_weights) in enumerate(
        zip(input_model_layer.get_weights(), new_layer.get_weights())
    ):
      if input_weights.shape == target_weights.shape:
        new_weights_to_assign[i] = input_weights
      else:
        logging.info(
            (
                'Source layer %s:%s has shape %s, '
                'target layer %s:%s has shape %s'
            ),
            input_model_layer.name,
            i,
            input_weights.shape,
            new_layer.name,
            i,
            target_weights.shape,
        )
        min_num_channels = min(input_weights.shape[2], target_weights.shape[2])
        new_weights_to_assign[i][:, :, :min_num_channels, :] = (
            input_weights[:, :, :min_num_channels, :]
        )

    model.layers[layer_i].set_weights(new_weights_to_assign)

  return model


def num_channels_from_checkpoint(filepath: str) -> int:
  """Determines the number of channels from a Keras checkpoint.

  Args:
    filepath: Path to the checkpoint.

  Returns:
    Number of channels inferred from the first convolutional kernel.

  Raises:
    ValueError: If the checkpoint format is not recognized or corresponds
      to an older DeepVariant architecture.
  """
  reader = tf.train.load_checkpoint(filepath)

  for name in reader.get_variable_to_shape_map().keys():
    if name.startswith(
        'layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE'
    ):
      weight_tensor = reader.get_tensor(name)
      return weight_tensor.shape[2]

    if name.startswith('layer_with_weights-0/layer_with_weights-0/kernel'):
      raise ValueError(
          'Detected an older DeepVariant Keras model architecture. '
          'Please regenerate the checkpoint with the new model.'
      )

  raise ValueError('Unexpected model format: could not infer channel count.')


def inceptionv3_with_imagenet(
    input_shape: Tuple[int, int, int],
) -> tf.keras.Model:
  """Builds an InceptionV3 backbone initialized from ImageNet (3 channels).

  This helper provides an InceptionV3 backbone with 3-channel ImageNet
  weights and a classification head compatible with DeepVariant. It is used
  to bootstrap models that operate on a different number of channels.

  Args:
    input_shape: Input shape (height, width, channels). The channel dimension
      is overridden to `3` inside this function.

  Returns:
    InceptionV3-based Keras model initialized with ImageNet weights.
  """
  input_shape = [input_shape[0], input_shape[1], 3]

  backbone = tf.keras.applications.InceptionV3(
      include_top=False,
      weights='imagenet',
      input_shape=input_shape,
      classes=dv_constants.NUM_CLASSES,
      pooling='avg',
  )

  weight_decay = _DEFAULT_WEIGHT_DECAY
  backbone_drop_rate = _DEFAULT_BACKBONE_DROPOUT_RATE

  features = tf.keras.layers.Dropout(backbone_drop_rate)(backbone.output)
  outputs = build_classification_head(features, l2=weight_decay)

  model = tf.keras.Model(
      inputs=backbone.input,
      outputs=outputs,
      name='inceptionv3',
  )
  model = add_l2_regularizers(model, tf.keras.layers.Conv2D, l2=weight_decay)

  return model


class LocalPatchTokenizer(tf.keras.layers.Layer):
  """Converts spatial feature maps into patch tokens.

  This layer applies a strided convolution to produce non-overlapping patches
  and then flattens them into a sequence of tokens.
  """

  def __init__(
      self,
      patch_size: int = 2,
      embed_dim: int = 256,
      **kwargs,
  ):
    super().__init__(**kwargs)
    self.patch_size = patch_size
    self.embed_dim = embed_dim
    self.proj = tf.keras.layers.Conv2D(
        filters=self.embed_dim,
        kernel_size=self.patch_size,
        strides=self.patch_size,
        padding='valid',
    )

  def call(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    x = self.proj(x)  # (B, H', W', C')
    shape = tf.shape(x)
    b = shape[0]
    h = shape[1]
    w = shape[2]
    c = shape[3]
    x = tf.reshape(x, [b, h * w, c])
    return x, h, w

  def get_config(self):
    config = super().get_config()
    config.update({
        'patch_size': self.patch_size,
        'embed_dim': self.embed_dim,
    })
    return config


class LocalSelfAttention(tf.keras.layers.Layer):
  """Windowed self-attention operating on local spatial neighborhoods."""

  def __init__(
      self,
      num_heads: int = 4,
      window_size: int = 4,
      **kwargs,
  ):
    super().__init__(**kwargs)
    self.num_heads = num_heads
    self.window_size = window_size

  def build(self, input_shape):
    c = input_shape[-1]
    self.qkv = tf.keras.layers.Dense(c * 3)
    self.proj = tf.keras.layers.Dense(c)

  def call(
      self,
      x: tf.Tensor,
      h: tf.Tensor,
      w: tf.Tensor,
  ) -> tf.Tensor:
    # x: (B, H*W, C)
    batch_size = tf.shape(x)[0]
    c = x.shape[-1]

    x_spatial = tf.reshape(x, [batch_size, h, w, c])

    window = self.window_size
    pad_h = (window - h % window) % window
    pad_w = (window - w % window) % window

    x_pad = tf.pad(x_spatial, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])
    hp = h + pad_h
    wp = w + pad_w

    x_win = tf.reshape(
        x_pad,
        [batch_size, hp // window, window, wp // window, window, c],
    )
    x_win = tf.transpose(x_win, [0, 1, 3, 2, 4, 5])
    x_win = tf.reshape(x_win, [batch_size, -1, window * window, c])

    qkv = self.qkv(x_win)
    q, k, v = tf.split(qkv, 3, axis=-1)

    attn = tf.matmul(q, k, transpose_b=True)
    attn = tf.nn.softmax(attn, axis=-1)

    out = tf.matmul(attn, v)
    out = self.proj(out)

    out = tf.reshape(
        out,
        [batch_size, hp // window, wp // window, window, window, c],
    )
    out = tf.transpose(out, [0, 1, 3, 2, 4, 5])
    out = tf.reshape(out, [batch_size, hp, wp, c])

    out = out[:, :h, :w, :]
    out = tf.reshape(out, [batch_size, h * w, c])

    return out

  def get_config(self):
    config = super().get_config()
    config.update({
        'num_heads': self.num_heads,
        'window_size': self.window_size,
    })
    return config


class HybridViTBlock(tf.keras.layers.Layer):
  """Local hybrid Transformer block over Inception feature maps."""

  def __init__(
      self,
      patch_size: int = 2,
      embed_dim: int = 256,
      num_heads: int = 4,
      window_size: int = 4,
      **kwargs,
  ):
    super().__init__(**kwargs)
    self.patch_size = patch_size
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.window_size = window_size

    self.tokenizer = LocalPatchTokenizer(
        patch_size=self.patch_size,
        embed_dim=self.embed_dim,
    )
    self.attn = LocalSelfAttention(
        num_heads=self.num_heads,
        window_size=self.window_size,
    )
    self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='ln_1')
    self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='ln_2')
    self.mlp = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(self.embed_dim * 4),
            tf.keras.layers.Dense(self.embed_dim),
        ],
        name='transformer_mlp',
    )

  def call(self, x: tf.Tensor) -> tf.Tensor:
    # x: (B, H, W, C)
    x_tokens, h, w = self.tokenizer(x)

    attn_out = self.attn(self.norm1(x_tokens), h=h, w=w)
    x_tokens = x_tokens + attn_out

    mlp_out = self.mlp(self.norm2(x_tokens))
    x_tokens = x_tokens + mlp_out

    x_out = tf.reshape(x_tokens, [-1, h, w, self.embed_dim])
    return x_out

  def get_config(self):
    config = super().get_config()
    config.update({
        'patch_size': self.patch_size,
        'embed_dim': self.embed_dim,
        'num_heads': self.num_heads,
        'window_size': self.window_size,
    })
    return config


def inceptionv3(
    input_shape: Tuple[int, int, int],
    weights: Optional[str] = None,
    init_backbone_with_imagenet: bool = True,
    config: Optional[ml_collections.ConfigDict] = None,
) -> tf.keras.Model:
  """Builds InceptionV3 backbone with a local HybridViT block after `mixed0`."""

  backbone = tf.keras.applications.InceptionV3(
      include_top=False,
      weights=None,
      input_shape=input_shape,
      classes=dv_constants.NUM_CLASSES,
      pooling=None,
  )

  try:
    stem_output = backbone.get_layer('mixed0').output
  except ValueError:
    stem_output = backbone.get_layer('mixed_0').output

  stem_channels = int(stem_output.shape[-1])

  vit_out = HybridViTBlock(
      patch_size=2,
      embed_dim=stem_channels,
      num_heads=4,
      window_size=4,
  )(stem_output)

  vit_resized = tf.image.resize(vit_out, tf.shape(stem_output)[1:3])
  merged = tf.keras.layers.Concatenate()([stem_output, vit_resized])

  x = tf.keras.layers.GlobalAveragePooling2D()(merged)

  if config and hasattr(config, 'backbone_dropout_rate'):
    drop_rate = config.backbone_dropout_rate
  else:
    drop_rate = _DEFAULT_BACKBONE_DROPOUT_RATE

  x = tf.keras.layers.Dropout(drop_rate)(x)
  outputs = build_classification_head(x, l2=_DEFAULT_WEIGHT_DECAY)

  model = tf.keras.Model(
      inputs=backbone.input,
      outputs=outputs,
      name='inceptionv3_localvit',
  )

  # Optionally compute FLOPs (best kept behind a config flag in production).
  try:
    from tensorflow.python.profiler.model_analyzer import profile
    from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

    @tf.function
    def model_forward(inputs):
      return model(inputs)

    concrete = model_forward.get_concrete_function(
        tf.TensorSpec(
            [1, input_shape[0], input_shape[1], input_shape[2]],
            tf.float32,
        )
    )
    profiler_options = ProfileOptionBuilder.float_operation()
    flops = profile(
        graph=concrete.graph,
        run_meta=tf.compat.v1.RunMetadata(),
        options=profiler_options,
    )
    logging.info('FLOPs for inceptionv3_localvit = %d.', flops.total_float_ops)
  except Exception as e:  # pylint: disable=broad-except
    logging.info('Could not compute FLOPs: %s', e)

  model = add_l2_regularizers(model, tf.keras.layers.Conv2D, l2=_DEFAULT_WEIGHT_DECAY)
  logging.info('Number of l2 regularizers: %s.', len(model.losses))

  if not weights and init_backbone_with_imagenet:
    logging.info('InceptionV3_LocalViT: init with ImageNet (3 channels).')
    model = load_weights_to_model_with_different_channels(
        model,
        inceptionv3_with_imagenet(input_shape),
    )
    return model

  if not weights and not init_backbone_with_imagenet:
    logging.info('InceptionV3_LocalViT: no initial checkpoint specified.')
    return model

  weights_num_channels = num_channels_from_checkpoint(weights)
  model_num_channels = input_shape[2]

  if weights_num_channels != model_num_channels:
    weights_input_shape = list(input_shape)
    weights_input_shape[2] = weights_num_channels

    input_model = inceptionv3(
        tuple(weights_input_shape),
        weights=None,
        init_backbone_with_imagenet=False,
        config=config,
    )
    logging.info(
        'Assigning weights from %s channels to %s channels.',
        weights_num_channels,
        model_num_channels,
    )
    model = load_weights_to_model_with_different_channels(model, input_model)
    return model

  logging.info('Loading weights from checkpoint: %s', weights)
  model.load_weights(weights)
  return model
