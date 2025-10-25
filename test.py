
import tensorflow as tf
# from tensorflow.keras import layers, models

import numpy as np


print("TensorFlow version:", tf.__version__)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print(x_train.shape[0])

# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(28, 28,1)),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dense(64, activation='relu'),
#   tf.keras.layers.Dense(10,activation='softmax')
# ])

new_model=True
learn=False
predict=False

if new_model:
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28,1)))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(10,activation='softmax'))
else: 
    model = tf.keras.models.load_model ('TOTOESTLA.keras')


if learn:
    model.compile(  optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
    
    history = model.fit(
        x_train,
        y_train,
        epochs=1)
    

if predict:
    a=model.predict(x_test)
    juste=0
    faux=0
    for i in range (len(a)):
        pos=np.argmax(a[i])
        res=y_test[i]
        if pos==res: juste+=1
        else: 
            faux+=1
            # print (pos,y_test[i])

    print (f'{juste} juste et {faux} faux')

a=x_test[10]
print (a)
for y in range (28):
    txt=""
    for x in range (28):
        if a[y,x]==0: txt+=" "
        else: txt+="O"
    print (txt)
print (y_test[10])
model.save ('TOTOESTLA.keras')




# del model
# model = tf.keras.models.load_model ('TOTOESTLA.keras')
# model.summary()





