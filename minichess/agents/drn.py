import tensorflow.keras as keras
import numpy as np
from tensorflow.keras import backend as K
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import tensorflow as tf

class DenseResidualNet:
    def __init__(self, input_shape, move_cap, init=True):
        if not init:
            return

        # Input layer
        position_input = keras.Input(input_shape)

        # Initial dense layer
        x = keras.layers.Dense(128, activation="relu")(position_input)

        # Dense residual blocks
        for _ in range(4):  # 4 residual blocks as an example
            shortcut = x
            x = keras.layers.Dense(128, activation="relu")(x)
            x = keras.layers.Dropout(0.3)(x)  # Regularization
            x = keras.layers.Dense(128, activation="relu")(x)
            x = keras.layers.Add()([x, shortcut])  # Residual connection

        # Policy head (move probabilities)
        policy = keras.layers.Conv2D(move_cap, (1, 1), activation="elu", padding="same")(x)
        policy = keras.layers.Flatten()(policy)
        policy_output = keras.layers.Softmax(name="policy_output")(policy)
        # Value head (board evaluation)
        val = keras.layers.Conv2D(16, (1, 1), name="value_conv", activation="elu", padding="same")(x)
        val = keras.layers.Flatten()(val)
        # val = keras.layers.Dense(256, activation="elu")(val)
        value_output = keras.layers.Dense(1, name="value_output", activation="tanh")(val)

        # Model that outputs both policy and value
        self.model = keras.Model(position_input, [policy_output, value_output])
        self.model.summary()
        self.model.compile(
            loss={"policy_output": keras.losses.CategoricalCrossentropy(), "value_output": keras.losses.MeanSquaredError()},
            loss_weights={"policy_output": 1.0, "value_output": 1.0},
            optimizer=keras.optimizers.Adam(learning_rate=0.001))

    def get_all_fc_layers_outputs(self, boards):
        """
        Returns the activations from all fully connected layers.
        The method will return the output of all hidden layers.
        """
        if len(boards.shape) == 3:
            boards = np.reshape(boards, (1, *boards.shape))

        # Input layer
        inp = self.model.input

        # Output of all fully connected layers (hidden layers)
        outputs = [layer.output for layer in self.model.layers if isinstance(layer, keras.layers.Dense)]

        # Create a function to get the activations
        functor = K.function([inp], outputs)

        BATCH_SIZE = 32
        all_layer_outs = []

        # Get the outputs for all boards in batches
        for i in tqdm(range(0, boards.shape[0], BATCH_SIZE)):
            layer_outs = functor([boards[i:i + BATCH_SIZE]])
            all_layer_outs.append(layer_outs)

        return all_layer_outs

    def fit(self, states, distributions, values, epochs=10):
        with tf.device('/gpu:0'):
            return self.model.fit(states, [distributions, values], epochs=epochs, batch_size=128)

    def predict(self, boards):
        if len(boards.shape) == 3:
            boards = np.reshape(boards, (1, *boards.shape))
        with tf.device('/cpu:0'):
            res = self.model(boards, training=False)
        policies, values = res
        return policies, values

    def predict_multi(self, boards):
        res = self.model.predict(np.array(boards))
        policies, values = res
        return policies, values