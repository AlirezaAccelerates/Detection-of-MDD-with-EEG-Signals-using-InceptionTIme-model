output_directory = ''                 #Put your directory
input_shape = [19, 61440]             #19 channels and 4-minute records
nb_classes = 2                        #MDD of Healthy
nb_filters = 64
verbose= True
use_residual = True
use_bottleneck = True
depth = 6
kernel_size = 41 - 1
callbacks = None
batch_size = 32
mini_batch_size = 32
bottleneck_size = 57
nb_epochs = 1500


def _inception_module( input_tensor, stride=1, activation='linear'):

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

def _shortcut_layer( input_tensor, out_tensor):
      shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                         padding='same', use_bias=False)(input_tensor)
      shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

      x = keras.layers.Add()([shortcut_y, out_tensor])
      x = keras.layers.Activation('relu')(x)
      return x

def build_model( input_shape, nb_classes):
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

      #reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

      file_path = output_directory + 'best_model.hdf5'

      model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                           save_best_only=True)

      callbacks = [reduce_lr, model_checkpoint]

      return model



model = build_model(input_shape, nb_classes)
model.summary()
model.save_weights(output_directory + 'model_init.hdf5')
hist = model.fit(X_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,  validation_split=0.15, shuffle=True,
                                  verbose=verbose, callbacks=callbacks)

model.save(output_directory + 'last_model.hdf5')

History = hist.history
losses = History['loss']
accuracies = History['accuracy']
