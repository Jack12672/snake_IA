
import tensorflow as tf
import numpy as np



print("TensorFlow version:", tf.__version__)




# 1. Données d'exemple
# -----------------------------
# 8 exemples, 6 features
X = np.array([
    #perdu, fruit, gauche, droite, haut, bas)
    [1, 0, 1, 0, 0, 0], #perdu
    [1, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 1],

    [0, 1, 1, 0, 0, 0], #gagné
    [0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 1]
], dtype=np.float32)

# Labels (classes 0 à 4) en one-hot
y = np.array([
    [0, 0, 0, 0.7, 0.7],  # classe 0 perdu vers gauche
    [0, 0, 0, 0.7, 0.7],  # classe 1 perdu vers droite
    [0, 0.7, 0.7, 0, 0],  # classe 2 perdu vers haut
    [0, 0.7, 0.7, 0, 0],  # classe 3 perdu vers bas

    [1, 0, 0, 0, 0],  # classe 4
    [1, 0, 0, 0, 0],  # classe 4
    [1, 0, 0, 0, 0],  # classe 4
    [1, 0, 0, 0, 0]  # classe 4
], dtype=np.float32)

new_model=False

# -----------------------------
# 2. Définition du modèle
# -----------------------------


if new_model:
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(6,)),      # 6 features
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')  # 5 classes
    ])
else:
    model = tf.keras.models.load_model ('GOD.keras')


# -----------------------------
# 3. Compilation
# -----------------------------
compil=False
if compil:
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',  # one-hot labels
        metrics=['accuracy']
    )

    # -----------------------------
    # 4. Entraînement
    # -----------------------------
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=200)
    history = model.fit(X, y,epochs=1, callbacks=[early_stop], verbose=0)

    model.save('GOD.keras')

    print(history.history)


test1=np.zeros((1,6),dtype=np.float32)
print (test1)

test = np.array([
    #perdu, fruit, gauche, droite, haut, bas)
    # [1, 0, 1, 0, 0, 0], #perdu
    # [1, 0, 0, 1, 0, 0],
    # [1, 0, 0, 0, 1, 0],
    # [1, 0, 0, 0, 0, 1],

    # [0, 1, 1, 0, 0, 0], #gagné
    # [0, 1, 0, 1, 0, 0],
    # [0, 1, 0, 0, 1, 0],
    [1, 0, 1, 0, 0, 0]
], dtype=np.float32)


print(test)
a=model.predict(test1,verbose=0)
print (a)
for i in range (len(a)):
    pos=np.argmax(a[i])
    print (pos)

exit()






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





