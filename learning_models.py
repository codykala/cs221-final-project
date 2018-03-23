""" 
Filename: learning_models.py
Date: 11/24/2017
Class: CS 221
Authors: Cody Kala, Brad Reyes, Manuel Torres Rojo
==================================================
Script for training our learning models on the SVHN dataset for CS221 Final Project.  
The script currently supports training the following learning models: 

	'lr', logistic regression model
	'svm', support vector machine
	'rf', random forest model

These models can be trained on the following features:

	'-b', binarized pixel values
	'-g', grayscale pixel values
	'-h', HOG feature values
	'-s', 0-mean, 1-std standardized pixel values
	'-m'. MinMax standardized pixel values
	'-sh', 0-mean, 1-std standardized HOG feature values
	'-mh', MinMax standardized HOG feature values

Execute the python script on the command line like so:

	python learning_models.py [classifier, -flag1, -flag2, ...]

As an example, to train a random forest model on HOG features, execute

	python learning_models.py rf -h

If no arguments are supplied or arguments are not given in the expected format, 
then the script defaults to training a logistic regression model on binarized 
pixel values, grayscale pixel values, and HOG feature values.  In other words, 
the default is equivalent to executing

	python learning_models.py lr -b -g -h

"""

# TODO: Play with SIFT, SURF, and ORB feature descriptors and see if they improve performance
# TODO: Look into Harris-Canny corner detection as feature descriptors
# TODO: Implement heat map generator function

# TODO: Use a trained CNN from ImageNet to generate a heat map for detecting which parts of an image are most important for detection/recognition
# TODO: Implement Convolution Neural Network (look at TensorFlow/PyTorch/Keras for doing this...)
# TODO: Look into early stopping, K-nearest neighbors
# TODO: How to use trained NN for feature extraction?
# TODO: Resolve overfitting for random forest model

""" Standard imports """
import sys
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.externals import joblib        			# Used to store our learned models to disk
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
import copy
import pickle
import os.path
import cv2                                  			# Used for SIFT and SURF feature sets

""" Directory paths """
MODEL_DIR_NAME = "models"       # Name of subdirectory where trained models are saved
CURVE_DIR_NAME = "curves"       # Name of subdirectory where learning curves are saved

""" Constants for learning models and features currently implemented. """
MODELS = ['lr', 'svm', 'rf', 'knn']
FEATURES = ['-b', '-g', '-h', '-m', '-s', '-mh', '-sh', '-sift']
DEFAULT_CLASSIFIER = 'lr'                   # Use this classifer when no classifier specified
DEFAULT_FEATURES = ['-b', '-g', '-h']       # Use these features when no features specified
DEFAULT_K = 10                              # For K-fold cross-validation
DEFAULT_LR_MAX_ITER = 10                    # Default maximum iterations for LR model
DEFAULT_SVM_MAX_ITER = 500                  # Default maximum iterations for SVM model
DEFAULT_RF_N_ESTIMATORS = 100                # Default number of estimators for RF model
DEFAULT_KNN_N_NEIGHBORS = 20 	            # Default number of neighbors for KNN model
IMG_PIXEL_DIM = 32                          # Pixel dimension of images in dataset
K_MEANS_SIZE = 100                           # Default value for SIFT/SURF w/ K-means features
OCCLUSION_SIZE = 8                          # Dimension for occlusion block

""" Constants for HOG feature descriptor """
WIN_SIZE = (32, 32)
BLOCK_SIZE = (16, 16)
BLOCK_STRIDE = (8, 8)
CELL_SIZE = (8, 8)
NBINS = 9

""" Feature extractor instantiations """
pss = StandardScaler()          # Standard scaler for pixel values
hss = StandardScaler()          # Standard scaler for HOG values
pms = MinMaxScaler()            # Minmax scaler for pixel values
hms = MinMaxScaler()            # Minmax scaler for HOG values
hog = cv2.HOGDescriptor(WIN_SIZE, BLOCK_SIZE, BLOCK_STRIDE, CELL_SIZE, NBINS)
sift = cv2.xfeatures2d.SIFT_create() 
bow = cv2.BOWKMeansTrainer(K_MEANS_SIZE)


def compute_k_means_features(sift_feature, cluster):
	kp, dsc = sift_feature
	buckets = [0] * K_MEANS_SIZE # initialization of buckets

	if len(kp) != 0:

		for i in range(len(dsc)):
			point = dsc[i]

			assigned_bucket_val = float("inf")
			assigned_bucket = None

			for j in range(K_MEANS_SIZE):
				cluster_centroid = cluster[j]
				dist = np.linalg.norm(point - cluster_centroid)

				if dist < assigned_bucket_val:
					assigned_bucket_val = dist
					assigned_bucket = j

			# now need to increment the bucket at this location by 1
			buckets[assigned_bucket] += 1

	# print buckets
	# print "---------------"
	return np.array(buckets)


def get_features(X, Y, included_features, training):
	""" Generates the features used in our learning algorithms.
		The following features are supported:
			'-g':  Grayscale values of pixels in each image
			'-b':  Binarized values of pixels in each image
			'-m':  MinMax standardized grayscale pixel values in each image
			'-s':  0-mean, 1-std standardized grayscale pixel values in each image
			'-h':  Histograms of oriented gradients features for each image
			'-sh': 0-mean, 1-std standardized Histograms of oriented gradients features for each image
			'-mh': MinMax standardized histograms of oriented gradients features for each image
			'-sift': SIFT feature descriptors w/ K-means clustering (histograms of bucket assignments used as features)
		Returns a numpy array of feature vectors for each image in |X|. """

	# print "Performing feature extraction..."

	# Iterate over the data to fit and/or transform the data as needed
	all_gray_images = []                # Images converted to grayscale
	all_gray_values = []                # Images converted to grayscale and flattened
	all_bin_values = []                 # Images converted to binarized and flattened
	all_hog_values = []                 # HOG features for each image
	all_sift_features = []              # SIFT (keypoints, descriptor) tuples for each image
	all_pixel_standard_values = []      # 0-1 standardized pixel values for each image 
	all_pixel_minmax_values = []        # Minmax standardized pixel values for each image
	all_hog_standard_values = []        # 0-1 HOG features for each image
	all_hog_minmax_values = []          # Minmax standardized HOG features for each image
	clusters = []                       

	for i in range(len(Y)):
		gray_values = cv2.cvtColor(X[i], cv2.COLOR_BGR2GRAY)   
		all_gray_images.append(gray_values)              
		all_gray_values.append(gray_values.flatten())

		if '-sift' in included_features:
			kp, dsc = sift.detectAndCompute(gray_values, None)
			if len(kp) != 0:
				bow.add(dsc)
			all_sift_features.append((kp, dsc))

		if '-b' in included_features:
			bin_values = cv2.adaptiveThreshold(gray_values, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 3)
			all_bin_values.append(bin_values.flatten())

		if '-h' in included_features or '-sh' in included_features or '-mh' in included_features:
			hog_values = hog.compute(gray_values.reshape((IMG_PIXEL_DIM, IMG_PIXEL_DIM)))
			all_hog_values.append(hog_values.flatten())

	if '-sift' in included_features:
		# print "clustering..."
		clusters = np.array(bow.cluster())
		# print "clustering begin"
		# print clusters
		# print clusters.shape
		# print "finished clustering..."

	# Only fit the scalers to the training data
	if training:
		if '-g' in included_features:
			pss.fit(all_gray_values)
			pms.fit(all_gray_values)

		if '-h' in included_features or '-sh' in included_features or '-mh' in included_features:
			hss.fit(all_hog_values)
			hms.fit(all_hog_values)

	if '-g' in included_features:
		all_pixel_standard_values = pss.transform(all_gray_values)
		all_pixel_minmax_values = pms.transform(all_gray_values)

	if '-h' in included_features or '-sh' in included_features or '-mh' in included_features:
		all_hog_standard_values = hss.transform(all_hog_values)
		all_hog_minmax_values = hms.transform(all_hog_values)

	# Do a second pass over the data to add the desired features to the feature set
	all_phis = []
	for i in range(len(Y)):
		# if i % 10000 == 0:
		# 	print "On image {}...".format(i)
		phi = []

		# Add grayscale values to phi 
		if '-g' in included_features:
		   phi.append(all_gray_values[i].flatten())
		# Add binarized features to phi 
		if '-b' in included_features:
			phi.append(all_bin_values[i].flatten())
		# Add standardized pixel values to phi 
		if '-s' in included_features:
			phi.append(all_pixel_standard_values[i].flatten())
		# Add minmax standardized pixel values to phi
		if '-m' in included_features:
			phi.append(all_pixel_minmax_values[i].flatten())
		# Add HOG features to phi
		if '-h' in included_features:
			phi.append(all_hog_values[i].flatten())
		# Add HOG features w/ minmax scaling to phi
		if '-mh' in included_features:
			phi.append(all_hog_minmax_values[i].flatten())
		# Add HOG features w/ standard scaling to phi
		if '-sh' in included_features:
			phi.append(all_hog_standard_values[i].flatten())
		# 
		if '-sift' in included_features:
			phi.append(compute_k_means_features(all_sift_features[i], clusters))

		phi = np.concatenate(tuple(phi))
		all_phis.append(phi) 
	
	return np.array(all_phis)


def train_with_cross_validation(classifier, features, Y, n = None, K = DEFAULT_K, verbose = False):
	""" Uses K-fold cross validation to predict the performance of the |classifier|
		on an independent dataset.  The |features| and |Y| values are split into
		|K| consecutive folds, and each fold is used once as a validation set 
		while the K - 1 remaining folds form the training set. """

	train_scores = []
	test_scores = []
	kf = KFold(n_splits = K)
	for train_indices, test_indices in kf.split(features, Y):
		train_features, test_features = features[train_indices], features[test_indices]
		train_Y, test_Y = Y[train_indices], Y[test_indices]
		model = train(classifier, train_features, train_Y, n)
		train_score = model.score(train_features, train_Y)
		train_scores.append(train_score)
		test_score = model.score(test_features, test_Y)
		test_scores.append(test_score)
	
	train_accuracy = sum(train_scores) / len(train_scores)
	test_accuracy = sum(test_scores) / len(test_scores)
	
	if verbose:
		print "================== RESULTS =================="
		print "{} model trained using {}-folds cross-validation.".format(classifier, K)
		print "Average accuracy on training set: {}".format(train_accuracy)
		print "Average accuracy on validation set: {}".format(test_accuracy)
		print "============================================="

	return train_accuracy, test_accuracy


def train(classifier, features, Y, n = None):
	""" Trains the |classifier| on the dataset using |features| and |Y|. 
		If a value for |n| is passed in, then it is used to set either 
		the |max_iter| or |n_estimator| parameters for the model, depending 
		on which |classifier| is specified. If no value for |n| is passed in, 
		then the model will use the default value for that parameter.
		Returns the trained |model|. """ 

	print "Training model..."
	model = None
	if classifier == 'lr':
		if n is not None:
			model = linear_model.LogisticRegression(max_iter = n)
		else:
			model = linear_model.LogisticRegression(max_iter = DEFAULT_LR_MAX_ITER)
	elif classifier == 'svm':
		if n is not None:
			model = LinearSVC(max_iter = n)
		else:
			model = LinearSVC(max_iter = DEFAULT_SVM_MAX_ITER)
	elif classifier == 'rf':
		if n is not None:
			model = RandomForestClassifier(n_estimators = n)
		else:
			model = RandomForestClassifier(n_estimators = DEFAULT_RF_N_ESTIMATORS)
	elif classifier == 'knn':
		if n is not None:
			model = KNeighborsClassifier(n_neighbors = n)
		else:
			model = KNeighborsClassifier(n_neighbors = DEFAULT_KNN_N_NEIGHBORS)

	model.fit(features, Y)
	print "Training complete!"
	return model


def get_learning_curve(classifier, included_features, features, Y, low, high):
	""" Plots the learning curve for the |classifier| using the given |features|
		and targets |Y|. The learning curve corresponds to the model's performance
		on the training set. 

		If the |classifier| is a logistic regression model or SVM, then the
		x-axis of the learning curve is the number of iterations.

		If the |classifier| is a random forest model, then the x-axis of the
		learning curve is the number of estimators.

		The values of |low| and |high| determine the range of x-axis values
		for the learning curve.
		""" 

	# Instead of using cross validation, we split |features| and |Y| into 
	# train and validation sets.  We portion off the last 10% of the data to use
	# for validation.
	PERCENT = 0.9
	train_features = features[:int(PERCENT * len(features))]
	train_Y = Y[:int(PERCENT * len(Y))]
	test_features = features[int(PERCENT * len(features)):]
	test_Y = Y[int(PERCENT * len(features)):]

	print "Generating learning curve for {}...".format(classifier)
	n_values = []
	train_scores = []
	test_scores = []
	for n in range(low, high + 1):
		print "On n = {}...".format(n)
		model = train(classifier, train_features, train_Y, n)
		train_score = model.score(train_features, train_Y)
		test_score = model.score(test_features, test_Y)
		n_values.append(n)
		train_scores.append(train_score)
		test_scores.append(test_score)
	
	train_plot = plt.scatter(n_values, train_scores, c = 'blue', marker = 'x')
	test_plot = plt.scatter(n_values, test_scores, c = 'red', marker = 'o')
	plt.legend((train_plot, test_plot), ('train learning curve', 'test learning curve'))
	plt.ylabel('Accuracy')
	if classifier == 'lr' or classifier == 'svm':
		plt.xlabel('Number of Iterations')
		plt.title('Accuracy vs. Number of Iterations for {} model'.format(classifier))
	elif classifier == 'rf':
		plt.xlabel('Number of estimators')
		plt.title('Accuracy vs. Number of Estimators for {} model'.format(classifier))
	
	filename = get_learning_curve_filename(classifier, included_features, low, high)
	plt.savefig(filename)
	print "...Done"


def get_learning_curve_filename(classifier, included_features, low, high):
	""" Generates the filename of the learning curve for the |classifier| with
		the |included_features|, where the parameter |n| takes on values between
		|low| and |high|, inclusive.  The filename has the following format:

			lc_[learning model]_[included features]_[low-high].png

		For example, if we generated a learning curve for a logistic regression
		model using binarized pixel features, grayscale pixel features, and HOG
		features for 1 to 25 iterations, then the filename would be

			lc_lr_-b-g-h_1-25.png """

	filename = "lc_{}_".format(classifier)
	for feature in included_features:
		filename = "{}{}".format(filename, feature)
	filename = "{}_{}-{}.png".format(filename, low, high)
	return filename


def parse_args():
	""" Reads arguments from the command line to determine which model to use and 
		which features to train on.  Returns the classifier and a list of features
		to use for training. """

	if len(sys.argv) == 1:
		classifier = DEFAULT_CLASSIFIER
		included_features = DEFAULT_FEATURES
	elif len(sys.argv) == 2:
		classifier = sys.argv[1]
		if classifier not in MODELS:
			print "Invalid classifier specified... using default classifier (lr)"
			classifier = 'lr'
		included_features = DEFAULT_FEATURES
	else:
		classifier = sys.argv[1]
		if classifier not in MODELS:
			print "Invalid classifier specified... using default classifier (lr)"
			classifier = 'lr'
		included_features = []
		for feature in sys.argv[2:]:
			if feature not in FEATURES:
				print "{} is not a valid feature, ignoring...".format(feature)
			elif feature in included_features:
				print "{} already included, ignoring...".format(feature)
			else:
				included_features.append(feature)
		if len(included_features) == 0:
			print "No valid features specified... using default features (-g, -b, -h)"
			included_features = DEFAULT_FEATURES
		else:
			included_features.sort()    # Sort for consistent filenames in get_learning_curve
	print(classifier, included_features)
	return classifier, included_features


def load_dataset(filename):
	""" Retrieves the dataset stored in the file given by |filename| in the current 
		directory.  Returns the images |X| and their labels |Y|. """

	dataset = scipy.io.loadmat(filename)
	X = dataset['X']
	Y = dataset['y']
	X = np.array([X[:,:,:,i] for i in range(len(Y))])
	Y = Y.flatten()
	return X, Y


def get_model_filename(classifier, included_features, n = 25):
	""" Returns the |filename| associated to |classifier| and |included_features|
		and |n| to be used to retrieve a saved model. """

	filename = "model_{}_".format(classifier)
	for feature in included_features:
		filename = "{}{}".format(filename, feature)
	filename = "{}_{}".format(filename, n)
	return filename


def save_model(model, classifier, included_features, n):
	""" Save the |model| to the disk with a name determined by |classifier|
		and |included_features|.  The model is stored as a text file in the
		subdirectory with name given by MODEL_DIR_NAME.  """ 

	if not os.path.isdir(MODEL_DIR_NAME):
		os.mkdir(MODEL_DIR_NAME)

	print "Saving model to disk..."
	file_path = os.path.dirname(os.path.abspath(__file__))
	file_path = os.path.join(file_path, MODEL_DIR_NAME)
	file_name = get_model_filename(classifier, included_features, n)
	file_path = os.path.join(file_path, file_name)
	file_object = open(file_path, 'wb')
	pickle.dump(model, file_object)
	file_object.close()
	print "Save successful!"


def load_model(file_name):
	""" Loads the |model| from the disk with the given |file_name|. 
		If no file with the given |file_name| is found, then None 
		is returned. """

	if not os.path.isdir(MODEL_DIR_NAME):
		print "Model directory not found, please try again."
		return None

	print "Loading model from disk..."
	file_path = os.path.dirname(os.path.abspath(__file__))
	file_path = os.path.join(file_path, MODEL_DIR_NAME)
	file_path = os.path.join(file_path, file_name)
	if os.path.isfile(file_path):
		file_object = open(file_path, 'r')
		model = pickle.load(file_object)
		print "Load successful!"
		return model
	else:
		print "No model found."
		return None


def generate_heat_map(orig_image, Y, included_features, model):
	""" This function is used to assess the importance of certain areas in
		the image to give the same classification as the model. 

		This function generates a heat map for the original image by
		occluding blocks of the image.  This altered image is fed to the
		model, producing the score.  If the model's score on the altered
		image differs from the model's score on the original image, then
		this suggests that the occluded region is important to the model's
		classification of the image.  """

	rows, cols, _ = orig_image.shape
	heatmap = [[(0, 0) for i in range(rows)] for j in range(cols)]
	orig_features = get_features([orig_image], [Y], included_features, False)
	orig_class = model.predict(orig_features)
	orig_scores = compute_scores(orig_features, included_features, model)

	i = 0
	j = 0

	while i + OCCLUSION_SIZE < rows:
		while j + OCCLUSION_SIZE < cols:
			copy_image = copy.deepcopy(orig_image)
			for k in range(OCCLUSION_SIZE):
				for l in range(OCCLUSION_SIZE):
					copy_image[i + k][j + l] = [0, 0, 0]
			copy_features = get_features([copy_image], [Y], included_features, False)
			copy_scores = compute_scores(copy_features, included_features, model)
			for k in range(OCCLUSION_SIZE):
				for l in range(OCCLUSION_SIZE):
					delta, iters = heatmap[i + k][j + l]
					heatmap[i + k][j + l] = (float(delta * iters + orig_scores[orig_class] - copy_scores[orig_class]) / (iters + 1), iters + 1)
					# if orig_class != copy_class:
					# 	heatmap[i + k][j + l] = (float(curr_score * curr_iters + 1) / (curr_iters + 1), curr_iters + 1) 
					# else:
					# 	heatmap[i + k][j + l] = (float(curr_score * curr_iters) / (curr_iters + 1), curr_iters + 1) 
			j += 1
		i += 1
		j = 0

	new_heatmap = [[0 for i in range(rows)] for j in range(cols)]
	for i in range(rows):
		for j in range(cols):
			delta, iters = heatmap[i][j]
			new_heatmap[i][j] = delta

	min_delta = min(min(new_heatmap))
	max_delta = max(max(new_heatmap))
	for i in range(rows):
		for j in range(cols):
			delta = new_heatmap[i][j]
			new_heatmap[i][j] = np.uint8(255 * float(delta - min_delta) / (max_delta - min_delta))

	new_heatmap = np.array(new_heatmap)	

	cv2.imshow('orig_img', orig_image)
	cv2.imwrite('orig_img.png', orig_image)
	cv2.waitKey(0)
	cv2.imshow('heatmap', new_heatmap)
	cv2.imwrite('heatmap.png', new_heatmap)
	cv2.waitKey(0)

def compute_scores(features, included_features, model):
	scores = np.matmul(model.coef_, features.transpose()) + model.intercept_.reshape((model.intercept_.size, 1))
	return scores


def main():
	# classifier, included_features = parse_args()
	# train_X, train_Y = load_dataset('train_32x32.mat')
	# train_features = get_features(train_X, train_Y, included_features, True)
	# get_learning_curve(classifier, included_features, train_features, train_Y, 1, 25)
	# train_with_cross_validation(classifier, train_features, train_Y)
	# model = train(classifier, train_features, train_Y)
	# save_model(model, classifier, included_features)
	

	classifier, included_features = parse_args()
	train_X, train_Y = load_dataset('train_32x32.mat')
	train_features = get_features(train_X, train_Y, included_features, True)
	filename = get_model_filename(classifier, included_features, n = 20)
	print filename
	model = load_model(filename)
	if model is None:
	    model = train(classifier, train_features, train_Y, n = 20)
	    save_model(model, classifier, included_features, n = 20)
	print "Training set error: {}".format(model.score(train_features, train_Y))
	# for i in range(100):
	# 	generate_heat_map(train_X[i], train_Y[i], included_features, model)
		# cv2.imwrite("image{}.png".format(i), train_X[i])



	# TODO: WHEN READY, EVALUATE PERFORMANCE ON THE TEST SET
	test_X, test_Y = load_dataset('test_32x32.mat')
	test_features = get_features(test_X, test_Y, included_features, False)
	print "Testing set error: {}".format(model.score(test_features, test_Y))

if __name__ == "__main__":
	main()

