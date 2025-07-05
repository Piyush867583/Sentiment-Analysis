import tensorflow as tf### models
import numpy as np### math computations
import matplotlib.pyplot as plt### plotting bar chart
import sklearn### machine learning library
import cv2## image processing
from sklearn.metrics import confusion_matrix, roc_curve### metrics
import seaborn as sns### visualizations
import datetime
import pathlib
import io
import os
import re
import string
import time
from numpy import random
import gensim.downloader as api
from PIL import Image
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import (Dense,Flatten,SimpleRNN,InputLayer,Conv1D,Bidirectional,GRU,LSTM,BatchNormalization,Dropout,Input,GlobalMaxPooling1D,Embedding,TextVectorization,LayerNormalization,MultiHeadAttention)
from tensorflow.keras.losses import BinaryCrossentropy,CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy,TopKCategoricalAccuracy, CategoricalAccuracy, SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from google.colab import drive
from google.colab import files
from tensorboard.plugins import projector

BATCH_SIZE=64

"""# Data Preparation"""

train_ds,val_ds,test_ds=tfds.load('imdb_reviews', split=['train', 'test[:50%]', 'test[50%:]'],as_supervised=True)

train_ds

for review,label in val_ds.take(2):
  print(review)
  print(label)

def standardization(input_data):
    '''
    Input: raw reviews
    output: standardized reviews
    '''
    lowercase=tf.strings.lower(input_data)
    no_tag=tf.strings.regex_replace(lowercase,"<[^>]+>","")
    output=tf.strings.regex_replace(no_tag,"[%s]"%re.escape(string.punctuation),"")

    return output

standardization(tf.constant("<u>In the movie?, </u>man called Tévèz, went to a friend’s pl**ce and they had a tensed discussion. I don’t love this movie! would you?<br> <br /><br />T"))

VOCAB_SIZE=10000
SEQUENCE_LENGTH=250
EMBEDDING_DIM=300

vectorize_layer=TextVectorization(
    standardize=standardization,
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=SEQUENCE_LENGTH
)

# lengths=[]
# words=[]

# for review,label in train_ds.take(100):
#   # for word in tf.strings.split(review, sep=" "):
#   #   if word in words:
#   #     pass
#   #   else:
#   #     words.append(word)
#   lengths.append(len(tf.strings.split(review, sep=" ")))

training_data=train_ds.map(lambda x,y:x)### input x and y and outputx
vectorize_layer.adapt(training_data)#### adapt the vectorize_layer to the training data

len(vectorize_layer.get_vocabulary())

def vectorizer(review,label):
    return vectorize_layer(review),label

train_dataset=train_ds.map(vectorizer)
val_dataset=val_ds.map(vectorizer)

vectorize_layer.get_vocabulary()[411]

for review,label in train_dataset.take(1):
  print(review)
  print(label)

train_dataset=train_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset=val_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

"""# Modeling

## Transformers

### Embeddings
"""

def positional_encoding(model_size,SEQUENCE_LENGTH):
  output=[]
  for pos in range(SEQUENCE_LENGTH):
    PE=np.zeros((model_size))
    for i in range(model_size):
      if i%2==0:
        PE[i]=np.sin(pos/(10000**(i/model_size)))
      else:
        PE[i]=np.cos(pos/(10000**((i-1)/model_size)))
    output.append(tf.expand_dims(PE,axis=0))
  out=tf.concat(output,axis=0)
  out=tf.expand_dims(out,axis=0)
  return tf.cast(out,dtype=tf.float32)

class Embeddings(Layer):
  def __init__(self, sequence_length, vocab_size, embed_dim,):
    super(Embeddings, self).__init__()
    self.token_embeddings=Embedding(
        input_dim=vocab_size, output_dim=embed_dim)
    self.sequence_length = sequence_length
    self.vocab_size = vocab_size
    self.embed_dim = embed_dim

  def call(self, inputs):
    embedded_tokens = self.token_embeddings(inputs)
    embedded_positions=positional_encoding(
        self.embed_dim,self.sequence_length)
    return embedded_tokens + embedded_positions

  def compute_mask(self, inputs, mask=None):
    return tf.math.not_equal(inputs, 0)

  def get_config(self):
      config = super().get_config()
      config.update({
        "sequence_length": self.sequence_length,
        "vocab_size": self.vocab_size,
        "embed_dim": self.embed_dim,
      })
      return config

test_input=tf.constant([[  2, 112,   10,   12,  5,   0,   0,   0,]])

emb=Embeddings(8,20000,256)
emb_out=emb(test_input)
print(emb_out.shape)

"""### Encoder"""

class TransformerEncoder(Layer):
    def __init__(self, embed_dim, dense_dim, num_heads,):
        super(TransformerEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim,
        )
        self.dense_proj=tf.keras.Sequential(
            [Dense(dense_dim, activation="relu"),Dense(embed_dim),]
        )
        self.layernorm_1 = LayerNormalization()
        self.layernorm_2 = LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None):
      if mask is not None:
        mask1 = mask[:, :, tf.newaxis]
        mask2 = mask[:,tf.newaxis, :]
        padding_mask = tf.cast(mask1&mask2, dtype="int32")

      attention_output = self.attention(
          query=inputs, key=inputs,value=inputs,attention_mask=padding_mask
      )

      proj_input = self.layernorm_1(inputs + attention_output)
      proj_output = self.dense_proj(proj_input)
      return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
      config = super().get_config()
      config.update({
        "embed_dim": self.embed_dim,
        "num_heads": self.num_heads,
        "dense_dim": self.dense_dim,
      })
      return config

encoder_outputs = TransformerEncoder(256,2048,2)(emb_out)
print(encoder_outputs.shape)

"""### Transformer Model"""

EMBEDDING_DIM=128
D_FF=1024
NUM_HEADS=8
NUM_LAYERS=1
NUM_EPOCHS=20

encoder_input=Input(shape=(None,), dtype="int64", name="input")
x = Embeddings(SEQUENCE_LENGTH,VOCAB_SIZE,EMBEDDING_DIM)(encoder_input)

for _ in range(NUM_LAYERS):
  x=TransformerEncoder(EMBEDDING_DIM,D_FF,NUM_HEADS)(x)

x = Flatten()(x)
output=Dense(1, activation="sigmoid")(x)

transformer = tf.keras.Model(
    encoder_input, output, name="transformer"
)
transformer.summary()

"""## Training"""

checkpoint_filepath = '/content/drive/MyDrive/nlp/sentiment_analysis/transformer.h5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

transformer.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history=transformer.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    callbacks=[model_checkpoint_callback])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model_loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('model_accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()



"""### Evaluation"""

transformer.load_weights(checkpoint_filepath)

test_dataset=test_ds.map(vectorizer)
test_dataset=test_dataset.batch(BATCH_SIZE)
transformer.evaluate(test_dataset)



"""# Testing"""

# test_pos="this movie looks very interesting, i love the fact that the actors do a great job in showing how people lived in the 18th century, which wasn't very good at all. But atleast this movie recreates this scenes! "
# test_neg="very good start, but movie started becoming boring at some point and unfortunately i didn't feel like this was properly produced as there was too much background noise, and the actors didn't look motivated at all "

test_data=tf.data.Dataset.from_tensor_slices([["this movie looks very interesting, i love the fact that the actors do a great job in showing how people lived in the 18th century, which wasn't very good at all. But atleast this movie recreates this scenes! "],
                                              ["very good start, but movie started becoming uninteresting at some point though initially i thought it would have been much more fun. There was too much background noise, so in all i didn't like this movie "],])

def vectorizer_test(review):
    return vectorize_layer(review)
test_dataset=test_data.map(vectorizer_test)

transformer.predict(test_dataset)



"""# LSH Attention"""

def look_one_back(x):
  x_extra=tf.concat([x[:,-1:,...],x[:,:-1,...]],axis=1)
  return tf.concat([x,x_extra],axis=2)

def sticker_look_one_back(x):
  x_extra=tf.concat([x[:-1:],x[:,:-1]],axis=1)
  return tf.concat([x,x_extra],axis=-1)

def causal_masker(a,b):
  a,b=tf.cast(a,dtype=tf.float32)+0.01,tf.cast(b,dtype=tf.float32)+0.01
  vals=tf.einsum('ipj,ipk->ipjk',b,1/a)
  out=tf.cast(tf.cast(tf.cast(vals,dtype=tf.int32),dtype=tf.bool),dtype=tf.int32)
  out=-out+1
  return tf.cast(out,dtype=tf.float32)

class LSHAttention(tf.keras.layers.Layer):
    def __init__(self,bucket_size=8,n_hashes=1):
        super(LSHAttention,self).__init__()
        self.n_hashes=n_hashes
        self.bucket_size=bucket_size

    def call(self,query,key,value,causal_masking=False):
        R=tf.random.normal((tf.shape(query)[0],tf.shape(query)[-1],self.bucket_size//2))
        xR=tf.matmul(query,R)
        concat_xR=tf.concat([xR,-xR],axis=-1)
        buckets=tf.math.argmax(concat_xR,axis=-1)

        sticker=tf.argsort(buckets)
        undo_sort=tf.argsort(sticker)
        sorted_query=tf.gather(query,sticker,axis=1,batch_dims=1)
        sorted_value=tf.gather(value,sticker,axis=1,batch_dims=1)

        chunked_query=tf.stack(tf.split(sorted_query,self.bucket_size,1),1)
        chunked_value=tf.stack(tf.split(sorted_value,self.bucket_size,1),1)

        sticker=tf.stack(tf.split(sticker,self.bucket_size,1),1)
        new_sticker=sticker_look_one_back(sticker)

        lb_chunked_query=look_one_back(chunked_query)
        lb_chunked_value=look_one_back(chunked_value)

        score=tf.einsum('bhie,bhje->bhij',chunked_query,lb_chunked_query)
        score/=tf.math.sqrt(tf.cast(query.shape[-1],tf.float32))

        if causal_masking==True:
            causal_mask=causal_masker(sticker,new_sticker)
            dots+=causal_mask*-1e-10
        score=tf.nn.softmax(score)
        output=tf.einsum('buij,buje->buie',score,lb_chunked_value)

        sorted_output=tf.reshape(output,(tf.shape(output)[0],tf.shape(query)[i],output.shape[3]))
        output=tf.gather(sorted_output,undo_sort,axis=1,batch_dims=1)
        return output
