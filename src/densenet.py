from tensorflow.keras import layers
from tensorflow.keras import backend

def DenseNet(
    blocks,
    input_tensor=None,
    input_shape=None,
    pooling=None,
	activation=lambda: layers.Activation('relu')):

  if input_tensor is None:
    img_input = layers.Input(shape=input_shape)
  else:
    if not backend.is_keras_tensor(input_tensor):
      img_input = layers.Input(tensor=input_tensor, shape=input_shape)
    else:
      img_input = input_tensor

  bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

  x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
  x = layers.Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
  x = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(
          x)
  # x = layers.Activation(activation, name='conv1/activation')(x)
  x = activation()(x)
  x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
  x = layers.MaxPooling2D(3, strides=2, name='pool1')(x)

  x = dense_block(x, blocks[0], name='conv2', activation=activation)
  x = transition_block(x, 0.5, name='pool2', activation=activation)
  x = dense_block(x, blocks[1], name='conv3', activation=activation)
  x = transition_block(x, 0.5, name='pool3', activation=activation)
  x = dense_block(x, blocks[2], name='conv4', activation=activation)
  x = transition_block(x, 0.5, name='pool4', activation=activation)
  x = dense_block(x, blocks[3], name='conv5', activation=activation)

  x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='bn')(x)
  # x = layers.Activation(activation, name='activation')(x)
  x = activation()(x)

  if pooling == 'avg':
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
  elif pooling == 'max':
    x = layers.GlobalMaxPooling2D(name='max_pool')(x)

  # Ensure that the model takes into account
  # any potential predecessors of `input_tensor`.
  if input_tensor is not None:
    inputs = layer_utils.get_source_inputs(input_tensor)
  else:
    inputs = img_input

  return inputs, x



def dense_block(x, blocks, name, activation):
  """A dense block.

  Arguments:
    x: input tensor.
    blocks: integer, the number of building blocks.
    name: string, block label.

  Returns:
    Output tensor for the block.
  """
  for i in range(blocks):
    x = conv_block(x, 32, name=name + '_block' + str(i + 1), activation=activation)
  return x


def transition_block(x, reduction, name, activation):
  """A transition block.

  Arguments:
    x: input tensor.
    reduction: float, compression rate at transition layers.
    name: string, block label.

  Returns:
    output tensor for the block.
  """
  bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
  x = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_bn')(
          x)
  # x = layers.Activation(activation, name=name + '_activation')(x)
  x = activation()(x)
  x = layers.Conv2D(
      int(backend.int_shape(x)[bn_axis] * reduction),
      1,
      use_bias=False,
      name=name + '_conv')(
          x)
  x = layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
  return x

def conv_block(x, growth_rate, name, activation):
  """A building block for a dense block.

  Arguments:
    x: input tensor.
    growth_rate: float, growth rate at dense layers.
    name: string, block label.

  Returns:
    Output tensor for the block.
  """
  bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
  x1 = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(
          x)
  # x1 = layers.Activation(activation, name=name + '_0_activation')(x1)
  x1 = activation()(x1)
  x1 = layers.Conv2D(
      4 * growth_rate, 1, use_bias=False, name=name + '_1_conv')(
          x1)
  x1 = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(
          x1)
  # x1 = layers.Activation(activation, name=name + '_1_activation')(x1)
  x1 = activation()(x1)
  x1 = layers.Conv2D(
      growth_rate, 3, padding='same', use_bias=False, name=name + '_2_conv')(
          x1)
  x = layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
  return x


def DenseNet121(input_tensor=None, input_shape=None, pooling=None, activation=lambda: layers.Activation('relu')):
  """Instantiates the Densenet121 architecture."""
  return DenseNet([6, 12, 24, 16], input_tensor, input_shape, pooling, activation)

def DenseNet169(input_tensor=None, input_shape=None, pooling=None, activation=lambda: layers.Activation('relu')):
  """Instantiates the Densenet169 architecture."""
  return DenseNet([6, 12, 32, 32], input_tensor, input_shape, pooling, activation)

def DenseNet201(input_tensor=None, input_shape=None, pooling=None, activation=lambda: layers.Activation('relu')):
  """Instantiates the Densenet201 architecture."""
  return DenseNet([6, 12, 48, 32], input_tensor, input_shape, pooling, activation)