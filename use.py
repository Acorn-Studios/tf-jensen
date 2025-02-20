import tensorflow as tf
from tensorflow import keras
import numpy as np
import gradio as gr

tokenizer = tf.keras.preprocessing.text.Tokenizer()

#Reads Text Inputs Here
f=open('Inputs.txt','r')
inputs = f.read().split('\n')
f.close()

corpus = inputs

tokenizer.fit_on_texts(corpus)
sequences = tokenizer.texts_to_sequences(corpus)

max_length = max([len(s) for s in sequences])

# Load your saved model
model = tf.keras.models.load_model('sentiment_mini-test')

def use(input_text):
  # Preprocess the input text
  sequences = tokenizer.texts_to_sequences([input_text])
  sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post', maxlen=max_length)

  # Make a prediction on the input text
  prediction = model.predict(sequences)[0]

  # Print the prediction
  return round(prediction[0])


iface = gr.Interface(fn=use, inputs="text", outputs="text")
iface.launch()
