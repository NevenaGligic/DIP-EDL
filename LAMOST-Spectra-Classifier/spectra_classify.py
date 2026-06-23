# if running on a server without display, uncomment the following line
# import matplotlib
# matplotlib.use('Agg')

# Standard libs
import numpy as np
import matplotlib.pyplot as plt

# Machine Learning
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics
from tensorflow import keras

# Local
from utils.functions import (
    load_fits_data, spectra_preview,
    energy_normalization, spectra_smooth, spectra_simulation,
    extract_spectral_features, extract_stat_features
)
from utils.CNN_model import CNN_Model_1D

'''
Use comment and uncomment to run specfic sections of the code as needed.
'''

# data loading and basic preprocessing
original_data_array = load_fits_data('train_data_01.fits')
normalized_data_array = energy_normalization(original_data_array)
smoothed_data_array = spectra_smooth(normalized_data_array, window_length=10, polyorder=3)

# data preview (plot & count)
spectra_preview(original_data_array, num_spectra_per_class=2, random_state=666)
counts = [np.sum(original_data_array[:, -1] == i) for i in range(3)]
print(f'Galaxy: {counts[0]} Quasar: {counts[1]} Star: {counts[2]}')

# data argumentation
# adjust the number of simulated spectra for each category as needed
simulated_data_galaxy = spectra_simulation(normalized_data_array, catagary_to_sim=0, number_to_sim=20)
simulated_data_quasar = spectra_simulation(normalized_data_array, catagary_to_sim=1, number_to_sim=20)
simulated_data_star = spectra_simulation(normalized_data_array, catagary_to_sim=2, number_to_sim=20)
new_data_array = np.vstack((simulated_data_galaxy, 
                            simulated_data_quasar, 
                            simulated_data_star))
new_smoothed_data_array = spectra_smooth(new_data_array, window_length=10, polyorder=3)
smoothed_data_array = np.vstack((smoothed_data_array, new_smoothed_data_array))

# extract spectral features: H-alpha, H-beta, H-gamma, H-delta, Mg, Na
line_list = [6562.81, 4861.34, 4340.47, 4101.75, 5183.62, 5889.95]
spectral_features_array = extract_spectral_features(smoothed_data_array, line_list, window=3)

# extract basic statistical features: mean, variance, max, min
stat_features_array = extract_stat_features(smoothed_data_array, n=50)

# PCA
pca_model = PCA(n_components=500, copy=True)
pca_features = pca_model.fit_transform(smoothed_data_array[:, :-1])

# check PCA results if needed
# plots available: Cumulative Explained Variance, PC1 vs PC2, PC1 histogram
plt.figure(figsize=(10,3), dpi=100)
label_map = {0: 'Galaxy', 1: 'Quasar', 2: 'Star'}

ax1 = plt.subplot(131)
ax1.plot(np.cumsum(pca_model.explained_variance_ratio_), linewidth=2)
ax1.set_xlabel('Number of PCs')
ax1.set_ylabel('Cumulative Explained Variance')

ax2 = plt.subplot(132)
for label in [0, 1, 2]:
    ax2.scatter(pca_features[smoothed_data_array[:, -1] == label, 0], 
                pca_features[smoothed_data_array[:, -1] == label, 1], 
                label=label_map[label], s=3, edgecolors='None')
ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')
ax2.legend(loc='upper right', fontsize=8)

ax3 = plt.subplot(133)
for label in [0, 1, 2]:
    ax3.hist(pca_features[smoothed_data_array[:, -1] == label, 0], 
             bins=50, histtype='step', label=label_map[label])
ax3.set_xlabel('PC1')
ax3.set_ylabel('Number')
ax3.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.show()

# final data preparation for training
# prepare for training
x = np.hstack((pca_features, spectral_features_array, stat_features_array))
y = smoothed_data_array[:, -1].astype(int)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=30)

# ------------------ Random Forest classifier ------------------
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=30, n_jobs=-1, verbose=1)
rf_model.fit(x_train, y_train)
y_predict_rf = rf_model.predict(x_test)

accuracy_rf = metrics.accuracy_score(y_test, y_predict_rf)
print('Random Forest Accuracy Score: %.3f' % (accuracy_rf))
f1_score_rf = metrics.f1_score(y_test, y_predict_rf, average='macro')
print('Random Forest F1 Score: %.3f' % (f1_score_rf))

confusion_matrix_rf = metrics.confusion_matrix(y_test, y_predict_rf, normalize='true')
disp = metrics.ConfusionMatrixDisplay(confusion_matrix_rf, display_labels=['Galaxy', 'Quasar', 'Star'])
disp.plot(cmap='Blues')
disp.im_.set_clim(vmin=0, vmax=1)
plt.title('Confusion Matrix - Random Forest')
plt.tight_layout()
plt.show()

# ------------------ Support Vector Machine classifier ------------------
svm_model = SVC(kernel='rbf', class_weight='balanced', random_state=30, verbose=True, probability=True)
svm_model.fit(x_train, y_train)
y_predict_svm = svm_model.predict(x_test)

accuracy_svm = metrics.accuracy_score(y_test, y_predict_svm)
print('SVM Accuracy Score: %.3f' % (accuracy_svm))
f1_score_svm = metrics.f1_score(y_test, y_predict_svm, average='macro')
print('SVM F1 Score: %.3f' % (f1_score_svm))

confusion_matrix_svm = metrics.confusion_matrix(y_test, y_predict_svm, normalize='true')
disp = metrics.ConfusionMatrixDisplay(confusion_matrix_svm, display_labels=['Galaxy', 'Quasar', 'Star'])
disp.plot(cmap='Blues')
disp.im_.set_clim(vmin=0, vmax=1)
plt.title('Confusion Matrix - SVM')
plt.tight_layout()
plt.show()

# ------------------ Convolutional Neural Network ------------------
# initialize CNN model
model = CNN_Model_1D(input_shape=x_train.shape[1], output_shape=3, conv_branch_num=3,
                     conv_kernel_size=[3, 5, 7, 9, 11, 13, 15, 17], node_num=[128, 64, 32])

# compile CNN model
model.compile(loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
              optimizer = keras.optimizers.Adam(learning_rate=0.0005),
              metrics =['accuracy'])

# plot flow chart if needed
keras.utils.plot_model(model, to_file='CNN_Model_1D.png', show_shapes=True)

# train CNN model
batch_size = 128
epochs = 20
training_history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

y_predict_cnn = model.predict(x_test)
y_predict_cnn = np.argmax(y_predict_cnn, axis=1)

accuracy_cnn = metrics.accuracy_score(y_test, y_predict_cnn)
print('CNN Accuracy Score: %.3f' % (accuracy_cnn))
f1_score_cnn = metrics.f1_score(y_test, y_predict_cnn, average='macro')
print('CNN F1 Score: %.3f' % (f1_score_cnn))

confusion_matrix_cnn = metrics.confusion_matrix(y_test, y_predict_cnn, normalize='true')
disp = metrics.ConfusionMatrixDisplay(confusion_matrix_cnn, display_labels=['Galaxy', 'Quasar', 'Star'])
disp.plot(cmap='Blues')
disp.im_.set_clim(vmin=0, vmax=1)
plt.title('Confusion Matrix - CNN')
plt.tight_layout()
plt.show()
