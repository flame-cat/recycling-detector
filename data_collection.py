from PIL import Image
import json
import pandas as pd

f = open('./train.json')
train = json.load(f)

id_to_file = {0: 'IMG_20181101_164908.jpg', 1: 'IMG_20181101_182204.jpg', 2: 'IMG_20181101_181546.jpg', 3: 'IMG_20181101_181111.jpg', 4: 'IMG_20181101_174004.jpg', 5: 'IMG_20181101_173213.jpg', 6: 'IMG_20181101_172743.jpg', 7: 'IMG_20181101_171520.jpg', 8: 'IMG_20181101_170727.jpg', 9: 'IMG_20181101_170541.jpg', 10: 'IMG_20181101_165826.jpg', 11: 'IMG_20181101_165424.jpg', 12: 'IMG_20181101_182646.jpg'}
category = {1: 'glass', 2: 'metal', 3: 'plastic'}

annotations = train['annotations']

data = {'filename': [], 'width': [], 'height': [], 'class': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [], 'image_id': []}

for a in annotations:
    # {'id': 745, 'image_id': 12, 'category_id': 3, 'bbox': [1063.0, 2459.0, 272.0, 400.0]}
    b = a['bbox']

    data['filename'].append(id_to_file[a['image_id']])
    data['width'].append(b[2])
    data['height'].append(b[3])
    data['class'].append(category[a['category_id']])
    data['xmin'].append(b[0])
    data['ymin'].append(b[1])
    data['xmax'].append(b[0] + b[2])
    data['ymax'].append(b[1] + b[3])
    data['image_id'].append(a['image_id'])

df = pd.DataFrame(data)

df.to_csv('./data/training/annotations/annotations.csv', index=False)
