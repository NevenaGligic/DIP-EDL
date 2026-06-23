# Machine Learning Classification of 1D Spectra in Astronomy

## Introduction & Motivation

Spectroscopy is one of the cornerstones of modern astrophysical research. In recent decade, large-scale spectroscopic surveys have produced extensive datasets, and revolutionized our understanding of the Milky Way and the Universe. The first step to analysis these spectra is usually to classify them into three major groups: galaxies, quasars, or stars.

Traditionally, the classification relies on template matching. However, this approach has limitations when processing low SNR spectra, spectra affected by instrumental noise, or objects exhibiting peculiar spectral features that do not match standard templates. This project explores automatic classification using data-driven machine learning algorithms (Random Forest, SVM, CNN), offering a potential alternative to improve classification accuracy and efficiency.

## Data

The Large Sky Area Multi-Object Fiber Spectroscopic Telescope (LAMOST) is a 4-meter meridian reflecting Schmidt telescope located in Hebei, China. With its ability of collecting 4,000 spectra simultaneously, LAMOST is one of the most efficient telescopes for spectral acquisition. 

A total of 100,000 good quality (SNR > 10) spectra are randomly selected from [LAMOST Data Release 9 (DR9)](http://lamost.org/dr9/) for training propose in this project. They are categorized into one of three types: galaxy, quasar, or star. All spectra have been evenly interpolated across the same wavelength range, from 3,900 Å to 9,000 Å, with 3,000 data points per spectrum. Data are stored in 10 FITS files, with each file containing 10,000 spectra. The structure of each FITS file is as follows: 

| No.  | Name    | Type        | Dimensions             | Format     |
| ---- | :------ | ----------- | ---------------------- | ---------- |
| 0    | PRIMARY | PrimaryHDU  | (3000, 10000)          | float64    |
| 1    |         | BinTableHDU | 10000 rows × 2 columns | ['K', 'K'] |

The first block `PRIMARY` contains the spectral data, where each row represents a single spectrum. The second block is a table with two columns: `objid` and `label`. Each `objid` corresponds to one spectrum, and the `label` indicates its type: 

| Type   | Label |
| ------ | ----- |
| Galaxy | 0     |
| Quasar | 1     |
| Star   | 2     |

Complete datasets for this project can be downloaded via Dropbox: [spectra_training_data.tar](https://www.dropbox.com/scl/fi/tp81mfopdqbep50vwhhhb/spectra_training_data.tar?rlkey=l2v9gr2dswpka2yk6gx9q9e6v&st=opdav8fs&dl=0). 

Example spectra of each type are shown below. Only one FITS file is used in the demo. If you want to preview one specific spectrum, simply refer to `spectra_preview.py` with your file name and index number. 

<div align=center>
<img src=".\figures\spectra_preview.png" width=55%>
</div>

## Repository Structure

```
LAMOST-Spectra-Classifier/
│
├── utils/
|   ├── __init__.py
│   ├── functions.py         # All data preparation and feature extraction functions
│   ├── CNN_model.py         # Multi-branch Conv1D CNN defined for the spectral classification
│
├── spectra_preview.py       # Make plot for one spectrum
├── spectra_classify.py      # Main pipeline: loading, feature extraction, training, classify
├── requirements.txt         # Python dependencies
├── README.md                # This file
```

## Workflow & Demo Results

### 1. Data Pre-processing

- **Energy Normalization:** Even within the same class of celestial objects, the absolute intensity of the spectra exhibit large differences in overall intensity due to their distances and brightness. To minimize the impact of energy magnitude differences in model training, each spectrum is normalized based on its total flux (energy) across the whole wavelength range. Such energy normalization would keep the original shape of spectrum. 
- **Savitzky-Golay Filter:** To reduce noise, we apply the Savitzky-Golay filter to spectra. It smooths a time series by fitting successive subsets of adjacent data points with a low-degree polynomial using least squares. After testing with a small dataset, we select a window size of 10 to ensure a balance between reducing noise and the loss of useful information. 

```python
original_data_array = load_fits_data('train_data_01.fits')
normalized_data_array = energy_normalization(original_data_array)
smoothed_data_array = spectra_smooth(normalized_data_array, window_length=10, polyorder=3)
```

### 2. Data Argumentation

In the training dataset, the distribution of the three classes is highly imbalanced. The overwhelming number of star samples can negatively impact the classification performance of the convolutional neural network. We generate simulated spectra by adding random normal distributed noise to the galaxy and quasar samples, effectively increasing their representation in the dataset. 

```python
simulated_data_galaxy = spectra_simulation(normalized_data_array, catagary_to_sim=0, number_to_sim=200)
simulated_data_quasar = spectra_simulation(normalized_data_array, catagary_to_sim=1, number_to_sim=200)
```
<div align=center>
<img src=".\figures\spectra_simulated2.png" width=55%>
</div>

### 3. Features Extraction

Spectral features and basic statistical features are extracted from spectra. The extracted features are combined with PCA results (refer to next subsection), and then utilized to train models. For spectral features, we calculate the mean flux around some important absorption lines, including Hα, Hβ, Hγ, Hδ, Mg, and Na. For statistical features, we divide the spectra into several blocks, and calculate the mean, variance, maximum, and minimum for each block. 

```python
line_list = [6562.81, 4861.34, 4340.47, 4101.75, 5183.62, 5889.95]
spectral_features_array = extract_spectral_features(smoothed_data_array, line_list, window=3)
stat_features_array = extract_stat_features(smoothed_data_array, n=150)
```

### 4. PCA

Principal Component Analysis (PCA) is applied to the data set in order to reduce dimension, which can help speed up the training and avoid overfitting. Each spectrum originally has 3,000 features (wavelength points). Our PCA keeps 500 principal components, achieving a 99% cumulative explained variance ratio. The following plots are made to inspect the performance of PCA. If the PCA works well, we may see clustering by class, even in just the first component. 

```python
pca_model = PCA(n_components=500, copy=True)
pca_features = pca_model.fit_transform(smoothed_data_array[:, :-1])
```
<div align=center>
<img src=".\figures\PCA.png" width=100%>
</div>

### 5. Classifier Training

PCA results and extracted features are used as the inputs to train classifier models. We test three different algorithms: Random Forest, Support Vector Machine (SVM), and Convolutional Neural Networks (CNN), and then compare their performance. Random Forest and SVM are directly called from `scikit-learn` package. 

```python
# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=30, n_jobs=-1, verbose=1)

# SVM
svm_model = SVC(kernel='rbf', class_weight='balanced', random_state=30, verbose=True, probability=True)

# CNN
model = CNN_Model_1D(input_shape=x_train.shape[1], output_shape=3, conv_branch_num=3,
                     conv_kernel_size=[3, 7, 9, 13, 17], node_num=[128, 64, 32])
model.compile(loss = losses.SparseCategoricalCrossentropy(from_logits=False), 
              optimizer = optimizers.Adam(learning_rate=0.0005), metrics =['accuracy'])
model.fit(x_train, y_train, batch_size=128, epochs=50, verbose=1)
```

We construct a 1D convolution neural network in this project. For the convolution layers, we apply multiple branches with difference size of convolution kernels in order to capture features at multiple scales of the spectra. The resulting feature maps are then concatenated into a one-dimensional vector, which is fed into a fully connected layer for classification. 

To address the issue of overfitting, we experimented with both dropout and batch normalization. We observed that the performance on the validation set was highly sensitive to the dropout rate. To reduce this complexity, we switch to batch normalization. Besides, there are 3 fully connected dense layers with 128, 64, 32 units respectively.  

<div align=center>
<img src=".\figures\CNN_Model_1D.png" width=55%>
</div>

```bash
Epoch 1/20
63/63 [==============================] - 20s 232ms/step - loss: 0.2853 - accuracy: 0.9183
Epoch 2/20
63/63 [==============================] - 16s 251ms/step - loss: 0.0942 - accuracy: 0.9744
Epoch 3/20
63/63 [==============================] - 16s 252ms/step - loss: 0.0549 - accuracy: 0.9870
Epoch 4/20
63/63 [==============================] - 18s 278ms/step - loss: 0.0369 - accuracy: 0.9908
Epoch 5/20
63/63 [==============================] - 19s 300ms/step - loss: 0.0185 - accuracy: 0.9966
```

The training of our CNN model takes about 20s per batch with 16 CPUs. The performances, results, and and confusion matrices of these three algorithms are summarized below. 

|                    | Random Forest | SVM   | CNN   |
| ------------------ | ------------- | ----- | ----- |
| **Accuracy Score** | 0.964         | 0.979 | 0.983 |
| **F1 Score**       | 0.936         | 0.957 | 0.971 |

<div align=center>
<img src=".\figures\confusion_matrices.png" width=100%>
</div>

## Quick Start

All scripts should be able to run on Windows, macOS, and Linux. The code is tested with Python 3.8.5 on personal laptop. These Python packages are required: `numpy`, `matplotlib`, `scipy`, `astropy`, `scikit-learn`, `tensorflow`. 

1. Clone the repository

```bash
git clone https://github.com/superdreamliner/LAMOST-Spectra-Classifier.git
cd LAMOST-Spectra-Classifier
```

2. Install dependencies if needed

```bash
pip install -r requirements.txt
```

3. Download and place training data into a local folder (not included in this repo). You can check the data sets and preview a specific spectrum by running

```bash
python spectra_preview.py
```

4. Run the main pipeline. Please refer to the workflow introduced in previous sections, and use comment and uncomment to run specific sections of the code as needed.

```bash
python spectra_classify.py
```

## References

[1] Bu, Y., Zeng, J., Lei, Z., & Yi, Z. (2019). Searching for Hot Subdwarf Stars from the LAMOST Spectra. III. Classification of Hot Subdwarf Stars in the Fourth Data Release of LAMOST Using a Deep Learning Method. *The Astrophysical Journal*, *886*(2), 128. 

[2] Sharma, K., Kembhavi, A., Kembhavi, A., Sivarani, T., Abraham, S., & Vaghmare, K. (2020). Application of convolutional neural networks for stellar spectral classification. *Monthly Notices of the Royal Astronomical Society*, *491*(2), 2280-2300.

## Acknowledgement & License

The Large Sky Area Multi-Object Fiber Spectroscopic Telescope (LAMOST) is a National Major Scientific Project built by the Chinese Academy of Sciences. Funding for the project has been provided by the National Development and Reform Commission. LAMOST is operated and managed by the National Astronomical Observatories, Chinese Academy of Sciences. 

This project is licensed under the MIT License. You are free to use, modify, and distribute the code for any purpose, with appropriate attribution to the author. 

## Contact

Created by Ruijie Shi as the final project for GLY 6932: Machine Learning in the Geosciences

Spring 2025, University of Florida

Email: ruijie.shi@ufl.edu
