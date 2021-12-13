This is a project for U of T CSC2515 ML course in Fall 2021 semester.

Run linear.py:
```
python linear.py --dataset cifar10 --model byol --aug None --dataset-size 20 --infer-new --C 1.778279387127403
```
New arguments added:

* --aug augments dataset with one of these augmentations: ['None', 'RandomCrop', 'RandomFlip', 'RandomRotation', 'ColourJitter']

* --dataset-size runs training and testing on <dataset-size> % of your train and test dataset.

* --ensemble generates pickles and csv files required for ensembling

* By default, if features already exist in results folder for a particular dataset and augmentation, 
linear.py will get these features and fit logReg classifier on them. 
If features don't exist, linear.py will infer features from SSL model (This can take time depending on --dataset-size). 

* If features exist in results folder and yet you want to re-infer them from SSL model, use --infer-new argument

* Barlow Twins: Just send --model arg as "barlow"

* Don't forget to rename aircraft dataset folder

* Download Cars dataset from https://ai.stanford.edu/~jkrause/cars/car_dataset.html (both test and train) and extract it in ../data/Cars



linear.py outputs:

* X_trainval_feature.pkl: size:(n_samples, n_features) Features inferred from our SSL model for train dataset.
* X_test_feature.pkl: size:(n_samples, n_features) Features inferred from our SSL model for test dataset.
* y_trainval.pkl: size:(n_samples, 1) groundtruth class labels for train dataset
* y_test.pkl: size:(n_samples, 1) groundtruth class labels for test dataset
* y_pred_labels: size:(n_samples, 1) class labels predicted by logistic regressor on test dataset
* y_pred_probabilities.pkl: size:(n_samples, num_classes) class probabilities predicted by logistic regressor on test dataset

Acknowledgements:

Parts of this code were borrowed from: [SSL-Transfer](https://github.com/linusericsson/ssl-transfer.git)