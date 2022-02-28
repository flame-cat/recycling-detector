from detecto.core import Model, Dataset
from detecto import utils, visualize

training = Dataset(label_data='./data/training/annotations/annotations.csv', image_folder='./data/training/images')
model = Model(['glass', 'metal', 'plastic'])

model.fit(training)

image = utils.read_image('./data/training/images/IMG_20181101_164908.jpg')
labels, boxes, scores = model.predict(image)
visualize.show_labeled_image(image, boxes, labels)
