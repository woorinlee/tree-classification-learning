import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import glob
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dense  
from tensorflow.keras.layers import BatchNormalization, Dropout, Flatten
from tensorflow.keras.optimizers import RMSprop, Adam

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

# 이미지 파일 이름으로 txt 파일 만들기
# filename treekinds(pine_tree/cypress) / pine_tree = 1, cypress = 0
def save_image_file_name():
    train_image_file_list = glob.glob('./train_data/*.jpg')
    f1 = open("train_data_list.txt", 'w')
    for i in range(1, len(train_image_file_list) + 1):
        file_name = os.path.splitext(os.path.basename(train_image_file_list[i-1]))[0]

        if "pine" in file_name:
            tree_kinds = 1
        else:
            tree_kinds = 0

        temp = str(file_name) + " " + str(tree_kinds)
        f1.write(temp + "\n")
    f1.close()
    print(">> train_data_list.txt 생성됨")

    test_image_file_list = glob.glob('./test_data/*.jpg')
    f2 = open("test_data_list.txt", 'w')
    for i in range(1, len(test_image_file_list) + 1):
        file_name = os.path.splitext(os.path.basename(test_image_file_list[i-1]))[0]

        if "pine" in file_name:
            tree_kinds = 1
        else:
            tree_kinds = 0

        temp = str(file_name) + " " + str(tree_kinds)
        f2.write(temp + "\n")
    f2.close()
    print(">> test_data_list.txt 생성됨")

def load_image_file_name(target_size= (224, 224), test_split_rate= 0.2):
    f1 = open("./train_data_list.txt")
    train_list_txt = f1.readlines()
    f1.close()
    
    # 파일 이름 섞기 
    np.random.shuffle(train_list_txt)
    dataset_1 = {"name": [], "label": [], "image": [ ]}
    
    for line in train_list_txt:
        image_name, species = line.split()
        image_file= "./train_data/"+ image_name + ".jpg"

        if os.path.exists(image_file):
            dataset_1["name"].append(image_name)
            dataset_1["label"].append(species) # pine_tree : 1, cypress: 0

            img = image.load_img(image_file, target_size=target_size)
            img = image.img_to_array(img)
            dataset_1["image"].append(img)

    f2 = open("./test_data_list.txt")
    test_list_txt = f2.readlines()
    f2.close()

    np.random.shuffle(test_list_txt)
    dataset_2 = {"name": [], "label": [], "image": [ ]}    
    for line in test_list_txt:
        image_name, species = line.split()
        image_file= "./test_data/"+ image_name + ".jpg"

        if os.path.exists(image_file):
            dataset_2["name"].append(image_name)
            dataset_2["label"].append(species) # pine_tree : 1, cypress: 0

            img = image.load_img(image_file, target_size=target_size)
            img = image.img_to_array(img)
            dataset_2["image"].append(img)
            
    train_dataset = {}
    train_dataset["image"] = np.array(dataset_1["image"])
    train_dataset["label"] = np.array(dataset_1["label"])
    train_dataset["name"]  = np.array(dataset_1["name"])
    print(">> train_list_txt dataset 생성됨")

    test_dataset = {}
    test_dataset["image"] = np.array(dataset_2["image"])
    test_dataset["label"] = np.array(dataset_2["label"])
    test_dataset["name"]  = np.array(dataset_2["name"])
    print(">> test_data_list dataset 생성됨\n")
    
    return train_dataset, test_dataset

save_image_file_name()

train_dataset, test_dataset = load_image_file_name()

x_train = train_dataset["image"]/255.0
y_train = train_dataset["label"]

x_test = test_dataset["image"]/255.0
y_test = test_dataset["label"]

y_train = tf.keras.utils.to_categorical(y_train) 
y_test = tf.keras.utils.to_categorical(y_test)

def create_cnn2d(input_shape, num_class = 2):
    inputs = Input(shape = input_shape)
    x= Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu')(inputs)
    x= BatchNormalization()(x)
    x= MaxPool2D()(x)

    x= Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu')(x)
    x= BatchNormalization()(x)
    x= MaxPool2D()(x)
    x= Dropout(rate = 0.5)(x)
    
    x = Flatten()(x)
    outputs = tf.keras.layers.Dense(units = num_class, activation = 'softmax')(x)
    model = tf.keras.Model(inputs = inputs, outputs = outputs)
    return model

model = create_cnn2d(input_shape = x_train.shape[1:])

print(">> 시작 시간 : " + str(datetime.datetime.now()) + "\n")

# 학습률 0.01 / Adam 최적화 사용
# opt = RMSprop(learning_rate = 0.001)
learning_rate = 0.01
epochs = 50
batch_size = 50

opt = Adam(learning_rate)
model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])
ret = model.fit(x_train, y_train, epochs, batch_size, verbose = 0)

train_loss, train_acc = model.evaluate(x_train, y_train, verbose = 2)
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose = 2)

print("\n>> Adam 최적화 사용")
print(">> learning_rate : " + str(learning_rate))
print(">> epochs : " + str(epochs))
print(">> batch_size : " + str(batch_size))

print("\n>> 종료 시간 : " + str(datetime.datetime.now()))

fig, ax = plt.subplots(1, 2, figsize = (10, 6))
ax[0].plot(ret.history['loss'], "g-")
ax[0].set_title("train loss")
ax[0].set_xlabel('epochs')
ax[0].set_ylabel('loss')

ax[1].plot(ret.history['accuracy'], "b-")
ax[1].set_title("accuracy")
ax[1].set_xlabel('epochs')
ax[1].set_ylabel('accuracy')
fig.tight_layout()
plt.show()
