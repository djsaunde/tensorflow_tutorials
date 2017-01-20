import tensorflow as tf
import numpy as np

# datasets
IRIS_TRAINING = 'iris_training.csv'
IRIS_TEST = 'iris_test.csv'

# load datasets
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TRAINING,
    target_dtype=np.int,
    features_dtype=np.float32)

test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TEST,
    target_dtype=np.int,
    features_dtype=np.float32)
    
# specify that all features have real-valued data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

# build a 3 layer MLP with 10, 20, and 10 units respectively
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[10, 20, 10], n_classes=3, model_dir='./models/iris_model')

# fit the model on the training dataset
classifier.fit(x=training_set.data, y=training_set.target, steps=2000)

# evaluate the accuracy of the model on the test dataset
accuracy_score = classifier.evaluate(x=test_set.data, y=test_set.target)['accuracy']
print('Accuracy: {0:f}'.format(accuracy_score))

# classify two new flower samples
new_samples = np.array(
    [[6.4, 3.2, 4.5, 1.5],
     [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
y = list(classifier.predict(new_samples, as_iterable=True))
print('Predictions: {}'.format(str(y)))
