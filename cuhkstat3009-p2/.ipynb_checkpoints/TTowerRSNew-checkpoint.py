import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

class TTowerRSNew(keras.Model):
    def __init__(self, n_user, n_item, n_genre, n_month, embedding_size, dense_size, embed_reg=1e-2, dense_reg=1e-2, fc_reg=1e-2, **kwargs):
        super(TTowerRSNew, self).__init__(**kwargs)
        self.num_users = n_user
        self.num_items = n_item
        self.num_genre = n_genre
        self.num_month = n_month
        self.embedding_size = embedding_size
        ## Embedding layers
        self.user_embedding = layers.Embedding(
            self.num_users,
            embedding_size,
            keras.initializers.he_normal(seed=None),
            embeddings_regularizer=keras.regularizers.l2(embed_reg),
        )
        self.item_embedding = layers.Embedding(
            self.num_items,
            embedding_size,
            keras.initializers.he_normal(seed=None),
            embeddings_regularizer=keras.regularizers.l2(embed_reg),
        )
        self.genre_embedding = layers.Embedding(
            self.num_genre,
            embedding_size,
            keras.initializers.he_normal(seed=None),
            embeddings_regularizer=keras.regularizers.l2(embed_reg),
        )
        self.month_embedding = layers.Embedding(
            self.num_month,
            embedding_size,
            keras.initializers.he_normal(seed=None),
            embeddings_regularizer=keras.regularizers.l2(embed_reg),
        )
        ##  Mapping layers
        self.user_dense = layers.Dense(dense_size, name='user_dense', activation='relu', 
                                       bias_regularizer=keras.regularizers.L2(dense_reg))
        self.item_dense = layers.Dense(dense_size, name='item_dense', activation='relu', 
                                       bias_regularizer=keras.regularizers.L2(dense_reg))
        self.network_dense = layers.Dense(dense_size, name='network_dense', activation='relu', 
                                       bias_regularizer=keras.regularizers.L2(dense_reg))
        self.joint_dense = layers.Dense(dense_size, name='joint_dense', activation='relu', 
                                       bias_regularizer=keras.regularizers.L2(dense_reg))
        self.fc1 = layers.Dense(round(dense_size/2), name='fc1', activation='relu', 
                                       bias_regularizer=keras.regularizers.L2(fc_reg))
        self.fc2 = layers.Dense(round(dense_size/4), name='fc2', activation='relu', 
                                       bias_regularizer=keras.regularizers.L2(fc_reg))
        self.fc3 = layers.Dense(round(dense_size/8), name='fc3', activation='relu', 
                                       bias_regularizer=keras.regularizers.L2(fc_reg))
        self.out_dense = layers.Dense(1, name='out_dense', activation='linear')

    def call(self, inputs):
        ## cate/cont data
        user_cont_feat, item_cont_feat, network_cont_feat, user_cate_feat, item_cate_feat = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]

        ## user embedding
        user_vec = self.user_embedding(user_cate_feat[:,0])
        month_vec = self.month_embedding(user_cate_feat[:,1])
        
        ## item embedding
        item_vec = self.item_embedding(item_cate_feat[:,0])
        genre_vec = self.genre_embedding(item_cate_feat[:,1])

        ## user/item dense mapping
        user_all_vec = layers.Concatenate()([user_cont_feat, user_vec, month_vec])
        item_all_vec = layers.Concatenate()([item_cont_feat, item_vec, genre_vec])

        user_dense_vec = self.user_dense(user_all_vec)
        item_dense_vec = self.item_dense(item_all_vec)
        network_dense_vec = self.network_dense(network_cont_feat)

        ## joint dense
        joint_vec = layers.Concatenate()([user_dense_vec, item_dense_vec, network_dense_vec])
        fc1_vec = self.joint_dense(joint_vec)
        fc2_vec = self.fc1(fc1_vec)
        fc3_vec = self.fc2(fc2_vec)
        out = self.out_dense(fc3_vec)
        return out
    
from sklearn.model_selection import KFold
import itertools
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class TTowerRSNew_CV(object):

	def __init__(self, n_user, n_item, n_genre, n_month, cv=3, 
                 embed_regs=[1e-4, 1e-3, 1e-2], 
                 dense_regs=[1e-4, 1e-3, 1e-2], 
                 fc_regs=[1e-4, 1e-3, 1e-2],
                 embedding_sizes=[100, 150, 200, 250, 300], 
                 dense_sizes=[100, 150, 200, 250, 300],
                 lrs=[1e-5, 1e-4, 1e-3, 1e-2], batches=[64,128,256]):
		self.n_user = n_user
		self.n_item = n_item
		self.n_genre = n_genre
		self.n_month = n_month
		self.cv = cv
		self.embed_regs = embed_regs
		self.dense_regs = dense_regs
		self.fc_regs = fc_regs
		self.embedding_sizes = embedding_sizes
		self.dense_sizes = dense_sizes
		self.lrs = lrs
		self.batches = batches
		self.best_model = {}
		self.cv_result = {'embedding_size': [], 'dense_size': [], 
                          'embed_reg': [], 'dense_reg': [], 'fc_reg': [], 
                          'lr': [], 'batch': [], 'train_rmse': [], 'valid_rmse': []}

	def grid_search(self, train_input, train_rating):
		## generate all combinations
		kf = KFold(n_splits=self.cv, shuffle=True)
		for (embedding_size, dense_size, embed_reg, dense_reg, fc_reg, lr, batch) in itertools.product(self.embedding_sizes, self.dense_sizes, self.embed_regs, self.dense_regs, self.fc_regs, self.lrs, self.batches):
			train_rmse_tmp, valid_rmse_tmp = 0., 0.
			for train_index, valid_index in kf.split(train_input[1]):
				# produce training/validation sets
				train_input_cv = []
				valid_input_cv = []
				for i in range(5):
					train_input_cv.append(train_input[i][train_index])
				train_rating_cv = train_rating[train_index]
				for i in range(5):
					valid_input_cv.append(train_input[i][valid_index])
				valid_rating_cv = train_rating[valid_index]
				# fit the model based on CV data
				model = TTowerRSNew(n_user=self.n_user, n_item=self.n_item, n_genre=self.n_genre, n_month=self.n_month, 
                                    embedding_size=embedding_size, dense_size=dense_size, 
                                    embed_reg=embed_reg, dense_reg=dense_reg, fc_reg=fc_reg)

				metrics = [keras.metrics.RootMeanSquaredError(name='rmse')]

				model.compile(optimizer=keras.optimizers.Adam(lr), 
                              loss=tf.keras.losses.MeanSquaredError(), metrics=metrics)

				callbacks = [keras.callbacks.EarlyStopping(monitor='val_rmse', min_delta=0, patience=10, verbose=1, 
                                                           mode='min', baseline=None, restore_best_weights=True)]

				history = model.fit(x=train_input_cv, y=train_rating_cv, batch_size=batch, epochs=150, verbose=2, callbacks=callbacks, validation_data=(valid_input_cv, valid_rating_cv))

				train_rmse_tmp_cv = history.history["rmse"][-1]
				valid_rmse_tmp_cv = history.history["val_rmse"][-1]
				train_rmse_tmp = train_rmse_tmp + train_rmse_tmp_cv / self.cv
				valid_rmse_tmp = valid_rmse_tmp + valid_rmse_tmp_cv / self.cv
				print('%d-Fold CV for embedding size: %d; dense size: %d; embed reg: %f; dense reg: %f; fc reg: %f; learning rate: %f; batch size: %d train_rmse: %f, valid_rmse: %f' 
						%(self.cv, embedding_size, dense_size, embed_reg, dense_reg, fc_reg, lr, batch, train_rmse_tmp_cv, valid_rmse_tmp_cv))
			self.cv_result['embedding_size'].append(embedding_size)
			self.cv_result['dense_size'].append(dense_size)
			self.cv_result['embed_reg'].append(embed_reg)
			self.cv_result['dense_reg'].append(dense_reg)
			self.cv_result['fc_reg'].append(fc_reg)
			self.cv_result['lr'].append(lr)
			self.cv_result['batch'].append(batch)
			self.cv_result['train_rmse'].append(train_rmse_tmp)
			self.cv_result['valid_rmse'].append(valid_rmse_tmp)
		self.cv_result = pd.DataFrame.from_dict(self.cv_result)
		best_ind = self.cv_result['valid_rmse'].argmin()
		self.best_model = self.cv_result.loc[best_ind]
	
	def plot_grid(self, data_source='valid'):
		sns.set_theme()
		if data_source == 'train':
			cv_pivot = self.cv_result.pivot("embedding_size", "dense_size", "embed_reg", "dense_reg", "fc_reg", "lr", "batch", "train_rmse")
		elif data_source == 'valid':
			cv_pivot = self.cv_result.pivot("embedding_size", "dense_size", "embed_reg", "dense_reg", "fc_reg", "lr", "batch", "valid_rmse")
		else:
			raise ValueError('data_source must be train or valid!')
		sns.heatmap(cv_pivot, annot=True, fmt=".3f", linewidths=.5, cmap="YlGnBu")
		plt.show()