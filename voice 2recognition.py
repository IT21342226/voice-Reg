

import os


dir = os.getcwd()



get_ipython().system('pip install jiwer')


# In[5]:


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from IPython import display
from jiwer import wer

dataPath = 'D:\\voice dataset'

wave_Path = dataPath+'\\wavs\\'
metadata_Path = dataPath+'\\metadata.csv'


import csv

meta_df = pd.read_csv(metadata_Path, header=None, quoting=csv.QUOTE_MINIMAL)


meta_df.tail()


# Assuming meta_df is your DataFrame
columns_to_remove = [0, 3, 4]
meta_df = meta_df.drop(columns=columns_to_remove)



meta_df = meta_df.drop(0)  # Drop the first row (index 0)


meta_df.columns=["file_name","normalized_transcription"]
meta_df = meta_df.sample(frac=1).reset_index(drop=True)
meta_df.head(3)


from sklearn.model_selection import train_test_split
# Split the data into train and test sets
train_df, test_df = train_test_split(meta_df, test_size=0.1, random_state=42, shuffle = True)


# Display the first 3 rows of the training set
print("Training set:")
print(train_df.head(3))

# Display the first 3 rows of the test set
print("\nTest set:")
print(test_df.head(3))

print(len(train_df))
print(len(test_df))


characters = [x for x in 'abcdefghijklmnopqrstuvwxyz,?! ']

char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")

num_to_char = keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)


frame_length = 256
frane_step = 160
fft_length = 384


def encode_single_sample(wav_file, label):
    file = tf.io.read_file(wave_Path+wav_file+'.wave')
    
    audio, _ = tf.audio.decode_wav(file)
    
    audio = tf.squeeze(audio, axis = -1)
    audio = tf.cast(audio, tf.float32)
    
    spectrogram = tf.signal.stft(audio, frame_length = frame_length, frame_step=frane_step, fft_length=fft_length)
    
    spectrogram = tf.abs(spectrogram)
    
    spectrogram = tf.math.pow(spectrogram,0.5)
    
    means  = tf.math.reduce_mean(spectrogram,1, keepdims=True)
    stddev = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram -means)/(stddev + 1e-10)
    
    label = tf.strings.lower(label)
    
    label = tf.strings.unicode_split(label, input_encoding="UTF-8")
    
    label = char_to_num(label)
    
    return spectrogram, label


batch_size = 32

train_dataset = tf.data.Dataset.from_tensor_slices(
    (list(train_df['file_name']), list(train_df['normalized_transcription'])))

train_dataset = (
    train_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(batch_size)
    .prefetch(buffer_size = tf.data.AUTOTUNE)
    )

test_dataset = tf.data.Dataset.from_tensor_slices(
    (list(test_df['file_name']), list(test_df['normalized_transcription']))
    )

test_dataset = (
    test_dataset.map(encode_single_sample,num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(batch_size)
    .prefetch(buffer_size = tf.data.AUTOTUNE))


fig = plt.figure(figsize=(8, 5))

for batch in train_dataset.take(1):
    spectrogram = batch[0][0].numpy()
    spectrogram = np.array([np.trim_zeros(x) for x in np.transpose(spectrogram)])
    label = batch[1][0]

    # Spectrogram
    label = tf.strings.reduce_join(tf.strings.unicode_split(tf.strings.unicode_encode(label, 'UTF-8'), ''), axis=-1).numpy().decode("utf-8")
    ax = plt.subplot(2, 1, 1)
    ax.imshow(spectrogram, vmax=1)
    ax.set_title(label)
    ax.axis("off")

    # Wav
    file_path = wave_Path + list(train_df["file_name"])[0] + ".wav"
    file = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(file)
    audio = audio.numpy()

    ax = plt.subplot(2, 1, 2)
    plt.plot(audio)
    ax.set_title("Signal Wave")
    ax.set_xlim(0, len(audio))
    display.display(display.Audio(np.transpose(audio), rate=16000))
    
plt.show()




