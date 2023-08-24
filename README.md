# tree-classification-learning(나무 분류 학습)

## 개요

나무(tree) 이미지를 학습하여 분류한다.

### 나무 이미지

|측백나무|소나무|
|:---:|:---:|
|<img width="70%" src="https://github.com/woorinlee/tree-classification-learning/assets/83910204/cbe1e153-9ee3-4440-927e-6d7a0b8b3708"/>|<img width="70%" src="https://github.com/woorinlee/tree-classification-learning/assets/83910204/a131e15a-e8b7-4ae3-aaa5-32b7d8471696"/>|

영상 학습 및 분류에 이용할 자료로 “측백나무”와 “소나무”를 선택하였다.

|나무 이미지 일부|
|:---:|
|<img width="100%" src="https://github.com/woorinlee/tree-classification-learning/assets/83910204/ca112591-2f03-446e-a73b-f130f8ee7041"/>|

전체 343장의 이미지 중 학습 데이터는 283장으로 전체의 82%, 테스트 데이터는 60장으로 전체의 17%의 비율이며 학습 데이터는 인터넷 웹 사이트에서 구한 공개 이미지, 테스트 데이터는 직접 촬영한 이미지를 사용하였다.

## 학습 과정

```
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
```

학습 전, 필요한 패키지들을 import한다.

```
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
```

glob 패키지를 통해 학습 데이터, 테스트 데이터 폴더 내의 파일 이름들을 리스트로 만들어 각각의 txt 형식 파일에 저장하는 함수 save_image_file_name를 작성한다.

```
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
```

학습 데이터, 테스트 데이터 파일 이름 리스트 텍스트 파일을 열고, 학습 데이터 이미지 파일들을 학습 데이터셋에, 테스트 데이터 이미지 파일들을 테스트 데이터셋에 저장한 후 반환하는 함수 load_image_file_name를 작성한다.

```
save_image_file_name()

train_dataset, test_dataset = load_image_file_name()

x_train = train_dataset["image"]/255.0
y_train = train_dataset["label"]

x_test = test_dataset["image"]/255.0
y_test = test_dataset["label"]

y_train = tf.keras.utils.to_categorical(y_train) 
y_test = tf.keras.utils.to_categorical(y_test)
```

위의 두 함수를 실행한 후, 학습 데이터, 테스트 데이터 변수에 각각의 데이터셋 정보를 입력한다.

```
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
```

input_shape = (224, 224, 3), num_class = 2의 모델을 생성하는 함수 create_cnn2d를 작성한다.

```
model = create_cnn2d(input_shape = x_train.shape[1:])

print(">> 시작 시간 : " + str(datetime.datetime.now()) + "\n")

# opt = RMSprop(learning_rate = 0.001)
opt = Adam(learning_rate)
learning_rate = 0.01
epochs = 50
batch_size = 50

model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])
ret = model.fit(x_train, y_train, epochs, batch_size, verbose = 0)

train_loss, train_acc = model.evaluate(x_train, y_train, verbose = 2)
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose = 2)

print("\n>> Adam 최적화 사용")
print(">> learning_rate : " + str(learning_rate))
print(">> epochs : " + str(epochs))
print(">> batch_size : " + str(batch_size))

print("\n>> 종료 시간 : " + str(datetime.datetime.now()))
```

위의 함수를 실행한 후 시간 측정을 위해 현재 시간을 출력한다. x_train 값과 y_train 값을 입력하고 learning_rate는 0.001, 최적화 알고리즘은 Adam, 손실 함수는 binary_crossentropy, epochs는 50, batch_size는 50의 값을 준 후 해당 모델을 학습한다.

```
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
```

loss 변화율과 accuracy 변화율을 출력하고 손실율, 정확도를 출력한다.

## 학습 결과

<img width="100%" src="https://github.com/woorinlee/tree-classification-learning/assets/83910204/e1592f5a-6af0-4706-967f-de6410877c34"/>

<img width="100%" src="https://github.com/woorinlee/tree-classification-learning/assets/83910204/e365c9cb-fabe-4f99-b524-5bc2409a0a50"/>

||정확도|
|:---|:---:|
|훈련 데이터|96.37%|
|테스트 데이터|66.67%|

## 결론

||학습 데이터 개수|테스트 데이터 개수|
|:---:|:---:|:---:|
|측백나무|58개|26개|
|소나무|225개|34개|

학습을 위한 이미지 데이터 개수가 위의 표와 같이 너무 적어서 제대로 된 결과가 나올수 없었다고 생각한다.