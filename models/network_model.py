import tensorflow as tf


def model(width, height):
    # define the input layer with the fixed dimension we used for processing images
    input_layer = tf.keras.layers.Input(shape=(height, width, 3))

    # encoding
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    # decoding
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)

    # define the output layer
    output_layer = tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    # create the model with the defines input and output layers
    model = tf.keras.models.Model(inputs=[input_layer], outputs=[output_layer])

    # compile the model with "adam" optimzer and "mse" loss
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    return model
