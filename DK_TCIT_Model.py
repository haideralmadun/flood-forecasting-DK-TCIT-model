import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, TimeDistributed, Dense, GlobalAveragePooling1D, LayerNormalization
from tensorflow.keras.optimizers import Nadam
from tcn import TCN  # Ensure the TCN library is installed
from custom_flood_loss import custom_flood_loss


# Self-Attention Layer
class ProbSparseSelfAttention(layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super(ProbSparseSelfAttention, self).__init__(**kwargs)
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        self.dense = layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q):
        batch_size = tf.shape(q)[0]
        q, k, v = self.wq(q), self.wk(k), self.wv(v)
        q, k, v = self.split_heads(q, batch_size), self.split_heads(k, batch_size), self.split_heads(v, batch_size)

        matmul_qk = tf.matmul(q, k, transpose_b=True)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(tf.cast(self.depth, tf.float32))
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)

        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(output, (batch_size, -1, self.d_model))
        return self.dense(concat_attention)

# Informer Block
class Informer(layers.Layer):
    def __init__(self, d_model, num_heads, conv_filters, **kwargs):
        super(Informer, self).__init__(**kwargs)
        self.self_attention = ProbSparseSelfAttention(d_model, num_heads)
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.dense_transform = layers.Dense(d_model)
        self.conv1 = layers.Conv1D(conv_filters, 3, padding='same', activation='relu')
        self.conv2 = layers.Conv1D(d_model, 3, padding='same')
        self.norm2 = LayerNormalization(epsilon=1e-6)

    def call(self, x):
        attn_output = self.self_attention(x, x, x)
        attn_output = self.norm1(attn_output + self.dense_transform(x))
        conv_output = self.conv2(self.conv1(attn_output))
        return self.norm2(conv_output + attn_output)

# DK-TCIT Model Definition
def DK_TCIT_Model(input_shape, n_steps_in, n_steps_out, d_model=128, num_heads=2, conv_filters=256, learning_rate=0.0005):
    input_layer = Input(shape=input_shape)

    # TimeDistributed CNN
    td_cnn = TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'))(input_layer)
    td_cnn = TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'))(td_cnn)
    td_cnn = TimeDistributed(MaxPooling1D(pool_size=1))(td_cnn)
    td_cnn = TimeDistributed(Flatten())(td_cnn)

    # Informer Module
    informer_output = Informer(d_model, num_heads, conv_filters)(td_cnn)
    informer_output = Informer(d_model, num_heads, conv_filters)(informer_output)

    # TCN Module
    tcn_output = TCN(nb_filters=64, kernel_size=3, dilations=[1, 2, 4, 8], return_sequences=True, activation='relu')(informer_output)
    tcn_output = TCN(nb_filters=64, kernel_size=3, dilations=[1, 2, 4, 8], return_sequences=False, activation='relu')(tcn_output)

    output_layer = Dense(n_steps_out)(tcn_output)
    model = Model(inputs=input_layer, outputs=output_layer)

    optimizer = Nadam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=custom_flood_loss)
    return model
