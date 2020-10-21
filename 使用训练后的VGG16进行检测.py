from keras.models import load_model
import numpy as np
from keras.preprocessing.image import load_img, img_to_array

label = np.array(['cat','dog','frog'])
# 载入模型
model = load_model('model_vgg16.h5')

# 导入图片
image = load_img('test1/225.jpg')

image.show()

image = image.resize((150,150))
image = img_to_array(image)
image = image/255
image = np.expand_dims(image,0)
image.shape
# (1, 150, 150, 3)
print(label[model.predict_classes(image)])
