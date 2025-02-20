import csv
import tensorflow as tf


# Step 1: Preprocess the text data

print('Converting Tokens...','\r')
tokenizer = tf.keras.preprocessing.text.Tokenizer()

# Load train.csv and set outputs and inputs
print('Loading Data...',end='')
corpus = []
outputs = []
with open('train.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        corpus.append(row['data'])
        outputs.append(int(row['ischeater'] == '1'))

tokenizer.fit_on_texts(corpus)
sequences = tokenizer.texts_to_sequences(corpus)

print('Encoding Tokens...',end='')
vocab_size = len(tokenizer.word_index) + 1
sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post')

print(' DONE')


# Step 2: Build the model

print('Define Model...',end='')

max_length = max([len(s) for s in sequences])

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, 8, input_length=max_length))
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(56, activation='relu'))

model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

print(' DONE')
print('Begining Compile Step...')

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()
input()


# Step 3: Train the model

print('Training...')

import numpy as np

x_train = sequences[:len(outputs)-1]
y_train = outputs[:len(outputs)-1]

x_train = np.array(x_train)
y_train = np.array(y_train)

# Train the model
model.fit(x_train, y_train, epochs=3)

model.save('jensen-vigil.h5')


# Step 4: Evaluate the model
print('Evaluating...')
x_test = sequences[len(outputs)-1:]
y_test = [0]

x_test = np.array(x_test)
y_test = np.array(y_test)

loss, accuracy = model.evaluate(x_test, y_test)

print('Loss:', loss)
print('Accuracy:', accuracy)