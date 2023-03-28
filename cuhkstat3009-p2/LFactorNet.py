import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

class LFactorNet(keras.Model):
    ## r_{ui} = p_u (users cont. month) @ q_i (item cont. genre) + mu_u + mu_i + mu_g + mu_m
    def __init__(self, num_users, num_items, num_genre, num_month,  embedding_size, reg, **kwargs):
        super(LFactorNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_items = num_items
        self.num_genre = num_genre
        self.num_month = num_month
        self.embedding_size = embedding_size
        # layers
        self.user_embedding = layers.Embedding(input_dim = num_users, 
                                               output_dim = embedding_size,
                                               embeddings_regularizer=keras.regularizers.l2(reg))
        self.user_bias = layers.Embedding(input_dim = num_users, output_dim=1)
        
        self.item_embedding = layers.Embedding(input_dim = num_items, 
                                               output_dim = embedding_size,
                                               embeddings_regularizer=keras.regularizers.l2(reg))
        self.item_bias = layers.Embedding(input_dim = num_items, output_dim=1)
        
        self.genre_embedding = layers.Embedding(input_dim = num_genre, 
                                               output_dim = embedding_size,
                                               embeddings_regularizer=keras.regularizers.l2(reg))
        self.genre_bias = layers.Embedding(input_dim = num_genre, output_dim=1)
        
        self.month_embedding = layers.Embedding(input_dim = num_month, 
                                               output_dim = embedding_size,
                                               embeddings_regularizer=keras.regularizers.l2(reg))
        self.month_bias = layers.Embedding(input_dim = num_month, output_dim=1)
        
    def call(self, inputs):
        user_vec = self.user_embedding(inputs[:,0])
        user_mu = self.user_bias(inputs[:,0])
        
        item_vec = self.item_embedding(inputs[:,1])
        item_mu = self.item_bias(inputs[:,1])
        
        genre_vec = self.genre_embedding(inputs[:,2])
        genre_mu = self.genre_bias(inputs[:,2])
        
        month_vec = self.month_embedding(inputs[:,3])
        month_mu = self.month_bias(inputs[:,3])
        
        user_all_vec = layers.Concatenate()([user_vec, month_vec])
        item_all_vec = layers.Concatenate()([item_vec, genre_vec])

        # tf.matmul(user_vec, tf.transpose(item_vec))
        # p_u @ q_i + mu_u + mu_i + mu_g + mu _m
        return tf.tensordot(user_all_vec, item_all_vec, 2) + user_mu + item_mu + genre_mu + month_mu