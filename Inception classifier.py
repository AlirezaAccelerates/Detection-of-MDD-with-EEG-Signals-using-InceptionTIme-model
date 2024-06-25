# Import necessary libraries
import keras
import numpy as np

# Define constants for model configuration
output_directory = '/output'  # Directory to save output files
input_shape = [19, 61440]  # Input shape: 19 channels and 4-minute records
nb_classes = 2  # Number of classes: MDD or Healthy
nb_filters = 64  # Number of filters in Conv layers
verbose = True  # Verbose output during training
use_residual = True  # Use residual connections
use_bottleneck = True  # Use bottleneck layers
depth = 6  # Depth of the model
kernel_size = 41 - 1  # Kernel size for Conv layers
callbacks = None  # Callbacks for training
batch_size = 32  # Batch size for training
mini_batch_size = 32  # Mini-batch size for training
bottleneck_size = 57  # Bottleneck size
nb_epochs = 1500  # Number of epochs for training

# Define the Inception module
def _inception_module(input_tensor, stride=1, activation='linear'):
    """
    Create an inception module.

    Args:
    input_tensor: Input tensor for the inception module.
    stride: Stride value for the convolutions (default is 1).
    activation: Activation function to use (default is 'linear').

    Returns:
    x: Output tensor after applying inception module.
    """
    if use_bottleneck and int(input_tensor.shape[-1]) > 1:
        input_inception = keras.layers.Conv1D(filters=bottleneck_size, kernel_size=1,
                                              padding='same', activation=activation, use_bias=False)(input_tensor)
    else:
        input_inception = input_tensor

    kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]

    conv_list = []

    for i in range(len(kernel_size_s)):
        conv_list.append(keras.layers.Conv1D(filters=nb_filters, kernel_size=kernel_size_s[i],
                                             strides=stride, padding='same', activation=activation, use_bias=False)(
            input_inception))

    max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

    conv_6 = keras.layers.Conv1D(filters=nb_filters, kernel_size=1,
                                 padding='same', activation=activation, use_bias=False)(max_pool_1)

    conv_list.append(conv_6)

    x = keras.layers.Concatenate(axis=2)(conv_list)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation='relu')(x)
    return x

# Define the shortcut layer for residual connections
def _shortcut_layer(input_tensor, out_tensor):
    """
    Create a shortcut (residual) connection between two tensors.

    Args:
    input_tensor: Input tensor to apply the shortcut to.
    out_tensor: Output tensor to add the shortcut to.

    Returns:
    x: Output tensor after adding the shortcut connection.
    """
    shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                     padding='same', use_bias=False)(input_tensor)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

    x = keras.layers.Add()([shortcut_y, out_tensor])
    x = keras.layers.Activation('relu')(x)
    return x

# Build the model using the defined Inception and shortcut layers
def build_model(input_shape, nb_classes):
    """
    Build the neural network model.

    Args:
    input_shape: Shape of the input data.
    nb_classes: Number of output classes.

    Returns:
    model: Compiled Keras model.
    """
    input_layer = keras.layers.Input(input_shape)

    x = input_layer
    input_res = input_layer

    for d in range(depth):
        x = _inception_module(x)

        if use_residual and d % 3 == 2:
            x = _shortcut_layer(input_res, x)
            input_res = x

    gap_layer = keras.layers.GlobalAveragePooling1D()(x)

    output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    model.compile(loss='mean_squared_error', optimizer=keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.01, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])

    # Reduce learning rate on plateau callback
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

    file_path = output_directory + '/best_model.hdf5'

    # Model checkpoint callback to save the best model
    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                       save_best_only=True)

    callbacks = [reduce_lr, model_checkpoint]

    return model

# Build and compile the model
model = build_model(input_shape, nb_classes)
model.summary()

# Save the initial weights of the model
model.save_weights(output_directory + '/model_init.hdf5')

# Train the model and save the training history
hist = model.fit(X_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs, validation_split=0.15, shuffle=True,
                 verbose=verbose, callbacks=callbacks)

# Save the final model
model.save(output_directory + '/last_model.hdf5')

# Extract and store training history
History = hist.history
losses = History['loss']
accuracies = History['accuracy']
