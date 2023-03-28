class glb_mean(object):
	def __init__(self):
		self.glb_mean = 0
	
	def fit(self, train_ratings):
		self.glb_mean = np.mean(train_ratings)
	
	def predict(self, test_tuple):
		pred = np.ones(len(test_tuple))
		pred = pred*self.glb_mean
		return pred
    
class user_mean(object):
	def __init__(self, n_user):
		self.n_user = n_user
		self.glb_mean = 0.
		self.user_mean = np.zeros(n_user)
	
	def fit(self, train_tuple, train_ratings):
		self.glb_mean = train_ratings.mean()
		for u in range(self.n_user):
			ind_train = np.where(train_tuple[:,0] == u)[0]
			if len(ind_train) == 0:
				self.user_mean[u] = self.glb_mean
			else:
				self.user_mean[u] = train_ratings[ind_train].mean()
	
	def predict(self, test_tuple):
		pred = np.ones(len(test_tuple))*self.glb_mean
		j = 0
		for row in test_tuple:
			user_tmp = row[0]
			pred[j] = self.user_mean[user_tmp]
			j = j + 1
		return pred
    
class item_mean(object):
	def __init__(self, n_item):
		self.n_item = n_item
		self.glb_mean = 0.
		self.item_mean = np.zeros(n_item)
	
	def fit(self, train_tuple, train_ratings):
		self.glb_mean = train_ratings.mean()
		for i in range(self.n_item):
			ind_train = np.where(train_tuple[:,1] == i)[0]
			if len(ind_train) == 0:
				self.item_mean[i] = self.glb_mean
			else:
				self.item_mean[i] = train_ratings[ind_train].mean()
	
	def predict(self, test_tuple):
		pred = np.ones(len(test_tuple))*self.glb_mean
		j = 0
		for row in test_tuple:
			item_tmp = row[1]
			pred[j] = self.item_mean[item_tmp]
			j = j + 1
		return pred
    
class genre_mean(object):
	def __init__(self, n_genre):
		self.n_genre = n_genre
		self.glb_mean = 0.
		self.genre_mean = np.zeros(n_genre)
	
	def fit(self, train_tuple, train_ratings):
		self.glb_mean = train_ratings.mean()
		for i in range(self.n_genre):
			ind_train = np.where(train_tuple[:,2] == i)[0]
			if len(ind_train) == 0:
				self.genre_mean[i] = self.glb_mean
			else:
				self.genre_mean[i] = train_ratings[ind_train].mean()
	
	def predict(self, test_pair):
		pred = np.ones(len(test_tuple))*self.glb_mean
		j = 0
		for row in test_pair:
			genre_tmp = row[2]
			pred[j] = self.genre_mean[genre_tmp]
			j = j + 1### Global Mean
		return pred
    
class month_mean(object):
	def __init__(self, n_month):
		self.n_month = n_month
		self.glb_mean = 0.
		self.month_mean = np.zeros(n_month)
	
	def fit(self, train_tuple, train_ratings):
		self.glb_mean = train_ratings.mean()
		for i in range(self.n_month):
			ind_train = np.where(train_tuple[:,3] == i)[0]
			if len(ind_train) == 0:
				self.month_mean[i] = self.glb_mean
			else:
				self.month_mean[i] = train_ratings[ind_train].mean()
	
	def predict(self, test_pair):
		pred = np.ones(len(test_tuple))*self.glb_mean
		j = 0
		for row in test_pair:
			month_tmp = row[3]
			pred[j] = self.month_mean[month_tmp]
			j = j + 1
		return pred