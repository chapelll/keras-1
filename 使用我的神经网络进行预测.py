from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np

label = np.array(['cat','dog','frog'])

# 载入模型
model = load_model('model_cnn.h5-frog1')

# 导入图片
image = load_img('test1/146.jpg')

# 展示图片
image.show()

image = image.resize((150,150))
image = img_to_array(image)
image = image/255
image = np.expand_dims(image,0)
image.shape
# (1, 150, 150, 3)

# 预测结果：
print(label[model.predict_classes(image)])