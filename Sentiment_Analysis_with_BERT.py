# Installation
"""

!pip install transformers datasets

"""# Imports"""

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
from tensorflow.keras.layers import Dense,Flatten,InputLayer,BatchNormalization,Dropout,Input,LayerNormalization
from tensorflow.keras.losses import BinaryCrossentropy,CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy,TopKCategoricalAccuracy, CategoricalAccuracy, SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from google.colab import drive
from google.colab import files
from datasets import load_dataset
from transformers import (BertTokenizerFast,TFBertTokenizer,BertTokenizer,RobertaTokenizerFast,
                          DataCollatorWithPadding,TFRobertaForSequenceClassification,TFBertForSequenceClassification,
                          TFBertModel,create_optimizer)

BATCH_SIZE=8

"""# Data Preparation for Bert Model"""

dataset_id='imdb'
dataset = load_dataset(dataset_id)

dataset

dataset['train'][0]

"""## Bert Model"""

model_id="bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(model_id)

tokenizer.is_fast

test_input_1='The Weather of Today is Gréat! zwp'
test_input_2='How are you doing?'
inputs=[test_input_1,test_input_2]

tokenizer.tokenize(inputs,)

output=tokenizer(inputs,padding=True,truncation=True,max_length=128)
print(output)

tokenizer.decode(output['input_ids'][0])

tokenizer.decode(output['input_ids'][1])

def preprocess_function(examples):
  return tokenizer(examples["text"],padding=True,truncation=True,)

tokenized_dataset = dataset.map(preprocess_function, batched=True)

tokenized_dataset

#tokenized_dataset['train'][0]

#data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

tf_train_dataset = tokenized_dataset["train"].to_tf_dataset(
    columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'],
    shuffle=True,
    batch_size=BATCH_SIZE,
    #collate_fn=data_collator
)

tf_val_dataset = tokenized_dataset["test"].to_tf_dataset(
    columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'],
    shuffle=True,
    batch_size=BATCH_SIZE,
    #collate_fn=data_collator
)

def swap_positions(dataset):
  return {'input_ids':dataset['input_ids'],
          'token_type_ids':dataset['token_type_ids'],
          'attention_mask':dataset['attention_mask'],},dataset['label']

tf_train_dataset=tf_train_dataset.map(swap_positions).prefetch(tf.data.AUTOTUNE)
tf_val_dataset=tf_val_dataset.map(swap_positions).prefetch(tf.data.AUTOTUNE)

for i in tf_train_dataset.take(1):
  print(i)

tf_val_dataset

"""## Data Preparation for Roberta Model"""

model_id="roberta-base"
tokenizer=RobertaTokenizerFast.from_pretrained(model_id)

def preprocess_function(examples):
  return tokenizer(examples["text"],padding=True,truncation=True,)

tokenized_dataset = dataset.map(preprocess_function,)# batched=True)

tokenized_dataset['train'][0]

tokenized_dataset

tf_train_dataset = tokenized_dataset["train"].to_tf_dataset(
    columns=['input_ids','attention_mask', 'label'],
    shuffle=True,
    batch_size=BATCH_SIZE,
    #collate_fn=data_collator
)

tf_val_dataset = tokenized_dataset["test"].to_tf_dataset(
    columns=['input_ids','attention_mask', 'label'],
    shuffle=True,
    batch_size=BATCH_SIZE,
    #collate_fn=data_collator
)

def swap_positions(dataset):
  return {'input_ids':dataset['input_ids'],
          'attention_mask':dataset['attention_mask'],},dataset['label']

tf_train_dataset=tf_train_dataset.map(swap_positions).prefetch(tf.data.AUTOTUNE)
tf_val_dataset=tf_val_dataset.map(swap_positions).prefetch(tf.data.AUTOTUNE)

for i in tf_train_dataset.take(1):
  print(i)

"""## Data Preparation for XtremeDistill Model"""

model_id="microsoft/xtremedistil-l6-h256-uncased"
tokenizer = BertTokenizerFast.from_pretrained(model_id)

tokenizer.is_fast

def preprocess_function(examples):
  return tokenizer(examples["text"],max_length=512,padding=True,truncation=True,)

tokenized_dataset = dataset.map(preprocess_function, batched=True)

tokenized_dataset

tf_train_dataset = tokenized_dataset["train"].to_tf_dataset(
    columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'],
    shuffle=True,
    batch_size=BATCH_SIZE,
    #collate_fn=data_collator
)

tf_val_dataset = tokenized_dataset["test"].to_tf_dataset(
    columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'],
    shuffle=True,
    batch_size=BATCH_SIZE,
    #collate_fn=data_collator
)

def swap_positions(dataset):
  return {'input_ids':dataset['input_ids'],
          'token_type_ids':dataset['token_type_ids'],
          'attention_mask':dataset['attention_mask'],},dataset['label']

tf_train_dataset=tf_train_dataset.map(swap_positions).prefetch(tf.data.AUTOTUNE)
tf_val_dataset=tf_val_dataset.map(swap_positions).prefetch(tf.data.AUTOTUNE)

for i in tf_val_dataset.take(1):
  print(i)

tf_val_dataset

"""# Modeling

## Based on TFBertForSequenceClassification
"""

model=TFBertForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=1)
model.summary()



"""## Based on XtremeDistillForSequenceClassification"""

model=TFBertForSequenceClassification.from_pretrained(model_id,num_labels=2)
model.summary()

"""## Based on TFBertModel"""

model=TFBertModel.from_pretrained("bert-base-uncased")
model.summary()

input_ids=Input(shape = (512,),dtype=tf.int64,name='input_ids')
token_type_ids=Input(shape = (512,),dtype=tf.int64,name='token_type_ids')
attention_mask=Input(shape = (512,),dtype=tf.int64,name='attention_mask')

x = model([input_ids,token_type_ids,attention_mask])
print(x)
x=Dense(128,activation='relu')(x[0][:,0,:])
output=Dense(1,activation='sigmoid',name='label')(x)

custom_bert = tf.keras.Model(inputs=[input_ids,token_type_ids,attention_mask], outputs=output)

custom_bert.summary()

"""## Based on TFRobertaForSequenceClassification"""

model=TFRobertaForSequenceClassification.from_pretrained(model_id,num_labels=2)
model.summary()



"""# Training"""

num_epochs = 3
batches_per_epoch = len(tokenized_dataset["train"]) // BATCH_SIZE
total_train_steps = int(batches_per_epoch * num_epochs)

optimizer, schedule = create_optimizer(init_lr=2e-5,num_warmup_steps=0, num_train_steps=total_train_steps)

model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=optimizer,
    metrics=['accuracy'],)
    #run_eagerly=True)

history=model.fit(
    tf_train_dataset.take(1000),
    validation_data=tf_val_dataset,
    epochs=3,)

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



"""# Testing

"""

inputs = tokenizer(["this movie looks very interesting, i love the fact that the actors do a great job in showing how people lived in the 18th century, which wasn't very good at all. But atleast this movie recreates this scenes! ",
                    "very good start, but movie started becoming uninteresting at some point though initially i thought it would have been much more fun. There was too much background noise, but later on towards the middle of the movie, my favorite character got in and he did a great job, so over "], padding=True,return_tensors="tf")

logits = model(**inputs).logits
print(logits)





"""# Conversion to Onnx Format

## Installation
"""

# !pip install -U tf2onnx
# !pip install onnxruntime

import onnxruntime as rt
import tf2onnx
rt.get_device()

"""## From Keras Model"""

output_path = "/content/drive/MyDrive/nlp/sentiment_analysis/xtremedistill.onnx"

spec = [tf.TensorSpec((None,512),tf.int64, name="input_ids"),
        tf.TensorSpec((None,512),tf.int64, name="token_type_ids"),
        tf.TensorSpec((None,512),tf.int64, name="attention_mask")]

model_proto, _ = tf2onnx.convert.from_keras(
    model, input_signature=spec,
    opset=17, output_path=output_path,)
output_names = [n.name for n in model_proto.graph.output]

print(output_names)

"""# Inference

## Benchmarking Onnx
"""

text=["this movie looks very interesting, i love the fact that the actors do a great job in showing how people lived in the 18th century, which wasn't very good at all. But atleast this movie recreates this scenes!"]

# text = ["this movie looks very interesting, i love the fact that the actors do a great job in showing how people lived in the 18th century, which wasn't very good at all. But atleast this movie recreates this scenes! ",
#                     "very good start, but movie started becoming uninteresting at some point though initially i thought it would have been much more fun. There was too much background noise, but later on towards the middle of the movie, my favorite character got in and he did a great job, so over ",
#                     "very good start, but movie started becoming uninteresting at some point though initially i thought it would have been much more fun. There was too much background noise, but later on towards the middle of the movie, my favorite character got in and he did a great job, so overall i will give this movie a pass "]


inputs = tokenizer(text,padding='max_length',max_length=512,truncation=True,return_tensors="np")

N_PREDICTIONS = 1
print(inputs)

providers=['CPUExecutionProvider']
m = rt.InferenceSession(output_path, providers=providers)

t1 = time.time()
for _ in range(N_PREDICTIONS):
  onnx_pred = m.run(["logits"], {'input_ids':inputs['input_ids'],
                                'token_type_ids':inputs['token_type_ids'],
                                'attention_mask':inputs['attention_mask']})
print("Time for a single Prediction", (time.time() - t1)/N_PREDICTIONS)

print(onnx_pred)

"""## Benchmarking TF"""

t1 = time.time()
for _ in range(N_PREDICTIONS):
  logits = model(**inputs).logits
print(logits)
print("Time for a single Prediction", (time.time() - t1)/N_PREDICTIONS)

tf, cpu = 600ms
tf, gpu = 130ms
tf_size = 50MB

onnx, cpu = 400ms
onnx, gpu = 8ms
onnx_size = 50MB
onnx_acc  = 91.9%

onnx_quantized, cpu = 190ms
onnx_quantized, gpu = 140ms
onnx_quantized_size = 13MB
onnx_quantized_acc  = 89.7%

"""# Quantization with Onnx"""

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = '/content/drive/MyDrive/nlp/sentiment_analysis/xtremedistill.onnx'
model_quant = '/content/drive/MyDrive/nlp/sentiment_analysis/xtremedistill_quantized.onnx'

quantized_model = quantize_dynamic(model_fp32, model_quant, weight_type = QuantType.QUInt8)

"""## Accuracy Drop due to Quantization"""

unbatched_val_dataset=tf_val_dataset.unbatch()

N_SAMPLES=1024

def accuracy(model):
  total=0
  for text,label in unbatched_val_dataset.take(N_SAMPLES):

    onnx_pred = model.run(["logits"], {'input_ids':[text['input_ids'].numpy()],
                                'token_type_ids':[text['token_type_ids'].numpy()],
                                'attention_mask':[text['attention_mask'].numpy()]})
    if np.argmax(onnx_pred, axis = -1)[0][0] == label.numpy():
      total+=1
  return (total/N_SAMPLES)*100

providers=['CPUExecutionProvider']
m = rt.InferenceSession(model_fp32, providers=providers)
m_q = rt.InferenceSession(model_quant, providers=providers)
print(accuracy(m_q))
print(accuracy(m))

"""# Understanding Temperature in Distillation"""

import numpy as np

def softmax(logits,T):
  denominator=np.sum([np.exp(i/T) for i in logits])
  return [np.exp(i/T)/denominator for i in logits]

logits=[10,13,17,5]

print("For T=1 ------>",softmax(logits,1))
print("For T=2 ------>",softmax(logits,2))
print("For T=3 ------>",softmax(logits,3))
print("For T=5 ------>",softmax(logits,5))
print("For T=10 ----->",softmax(logits,10))
print("For T=10000 -->",softmax(logits,10000))

