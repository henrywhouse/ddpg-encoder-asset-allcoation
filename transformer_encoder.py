#############################
###  Transformer Encoder  ###
#############################

from keras.layers import GRU, LayerNormalization, Dropout, MultiHeadAttention, Layer, InputLayer, Flatten

class EncoderLayer(Layer):
    """Transformer encoder layer that uses GRU layer instead of feed-forward; returns a sequence
    """
    def __init__(self, input_shape, head_size, num_heads, gru_units, dense_units, dropout=0.0):
        """Initialize the Encoder Layer"""
        
        super(EncoderLayer, self).__init__()
        
        self.head_size = head_size
        self.num_heads = num_heads
        self.gru_units = gru_units
        self.dense_units = dense_units
        self.dropout = dropout

        self.input_layer = InputLayer(input_shape=input_shape)
        self.flatten = Flatten()
        self.layer_norm1 = LayerNormalization(epsilon=1e-6)
        self.multi_head_attention = MultiHeadAttention(
            key_dim=self.head_size, num_heads=self.num_heads, dropout=self.dropout)
        self.dropout1 = Dropout(self.dropout)
        self.layer_norm2 = LayerNormalization(epsilon=1e-6)
        self.gru = GRU(self.gru_units, activation="relu", return_sequences=True)
        self.dropout2 = Dropout(self.dropout)


    def call(self, inputs):
        """Forward pass through the layer"""

        x = self.layer_norm1(inputs)
        x = self.multi_head_attention(x, x)
        x = self.dropout1(x)
        res = x + inputs
        x = self.layer_norm2(res)
        x = self.gru(x)
        x = self.dropout2(x)
        return x
