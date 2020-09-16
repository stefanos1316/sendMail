import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = models.model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)
