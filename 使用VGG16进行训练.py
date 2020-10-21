from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np

# vgg16去掉全连接层模型，include_top: 表示全连接层
vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3))
vgg16_model.output_shape
# (None, 4, 4, 512)

# 搭建新的全连接层
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(3, activation='softmax'))

model = Sequential()
model.add(vgg16_model)
model.add(top_model)

# 图片生成器
train_datagen = ImageDataGenerator(
    rotation_range = 40,     # 随机旋转度数
    width_shift_range = 0.2, # 随机水平平移
    height_shift_range = 0.2,# 随机竖直平移
    rescale = 1/255,         # 数据归一化
    shear_range = 20,       # 随机错切变换
    zoom_range = 0.2,        # 随机放大
    horizontal_flip = True,  # 水平翻转
    fill_mode = 'nearest',   # 填充方式
)
test_datagen = ImageDataGenerator(
    rescale = 1/255,         # 数据归一化
)

# 批次大小
batch_size = 32

# 生成训练数据
train_generator = train_datagen.flow_from_directory(
    'train1',
    target_size=(150,150),
    batch_size=batch_size,
)

# 测试数据
test_generator = test_datagen.flow_from_directory(
    'test1',
    target_size=(150,150),
    batch_size=batch_size,
)
# Found 400 images belonging to 2 classes.
# Found 200 images belonging to 2 classes.

# 目标索引
train_generator.class_indices
# {'cat': 0, 'dog': 1}

# 定义优化器， 代价函数，训练过程中计算准确率
model.compile(optimizer=SGD(lr=1e-4, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

# 训练
model.fit_generator(train_generator, steps_per_epoch=len(train_generator), epochs=20, validation_data=test_generator, validation_steps=len(test_generator))

# 模型保存
model.save('model_vgg16.h5')