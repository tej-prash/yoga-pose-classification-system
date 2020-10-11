# -*- coding: utf-8 -*-

from google.colab import drive
drive.mount('/gdrive/')

"""Index

1.   Loading data from google drive.
2.   Preprocess data - Interpolating with mean
3.   LSTM to classify poses.
4.   VAE for pose generation - 2 classes (Tensorflow)
5.   VAE with pose generation - 2 classes (Keras)
6.   Latent space interpolation and plotting
"""

# Preprocess data
import json
from os import listdir
from os.path import isfile, join
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import pandas as pd

dataset=[]
y=[]
# Read data from json files
# 1) Plank pose data
# Enter list of paths that need to be read
paths=[]
for path in paths:
  files = [f for f in listdir(path) if isfile(join(path, f))]
  for file_name in files:
    with open(path+file_name,'r') as fp:
      json_data=json.load(fp)
      np_data=np.array(json_data["people"][0]["pose_keypoints_2d"])
      # print(np_data)
      np_trimmed_data=[np_data[i] for i in range(np_data.shape[0]) if (i+1)%3!=0]
      np_trimmed_data=np.array(np_trimmed_data)
      np_trimmed_data=np_trimmed_data.reshape((25,2))
      dataset.append(np_trimmed_data)
y.extend([[1] for _ in range(len(dataset))])
print("Done with plank pose")

# 2) Mountain pose data
# Enter list of paths that need to be read
paths=[]
for path in paths:
  files = [f for f in listdir(path) if isfile(join(path, f))]
  for file_name in files:
    with open(path+file_name,'r') as fp:
      json_data=json.load(fp)
      np_data=np.array(json_data["people"][0]["pose_keypoints_2d"])
      np_trimmed_data=[np_data[i] for i in range(np_data.shape[0]) if (i+1)%3!=0]
      np_trimmed_data=np.array(np_trimmed_data)
      np_trimmed_data=np_trimmed_data.reshape((25,2))
      dataset.append(np_trimmed_data)
y.extend([[0] for _ in range(len(dataset)-len(y))])
print("Done with mountain pose")

# 3) Tree pose data
invalid_list=[]
# Enter list of paths that need to be read
paths=[]
for path in paths:
  files = [f for f in listdir(path) if isfile(join(path, f))]
  for file_name in files:
    with open(path+file_name,'r') as fp:
      json_data=json.load(fp)
      try:
        np_data=np.array(json_data["people"][0]["pose_keypoints_2d"])
      except:
        invalid_list.append(path+file_name)
        continue
      np_trimmed_data=[np_data[i] for i in range(np_data.shape[0]) if (i+1)%3!=0]
      np_trimmed_data=np.array(np_trimmed_data)
      np_trimmed_data=np_trimmed_data.reshape((25,2))
      dataset.append(np_trimmed_data)
y.extend([[2] for _ in range(len(dataset)-len(y))])

print("Done with tree pose")

# 4) Warrior pose data
# Enter list of paths that need to be read
paths=[]
for path in paths:
  files = [f for f in listdir(path) if isfile(join(path, f))]
  for file_name in files:
    with open(path+file_name,'r') as fp:
      json_data=json.load(fp)
      try:
        np_data=np.array(json_data["people"][0]["pose_keypoints_2d"])
      except:
        invalid_list.append(path+file_name)
        continue
      np_trimmed_data=[np_data[i] for i in range(np_data.shape[0]) if (i+1)%3!=0]
      np_trimmed_data=np.array(np_trimmed_data)
      np_trimmed_data=np_trimmed_data.reshape((25,2))
      dataset.append(np_trimmed_data)
y.extend([[3] for _ in range(len(dataset)-len(y))])

print("Done with warrior pose")

# print(invalid_list)
# print(len(invalid_list))

y=np.array(y)
print(y.shape)
dataset=np.array(dataset)

# Splitting the dataset
X=dataset

print(X.shape)


# Fill in outliers
df_trimmed_1 = pd.DataFrame(data=X[:,:,0],columns=np.arange(X.shape[1]))
df_trimmed_1['label']=y[:,0]

df_trimmed_2 = pd.DataFrame(data=X[:,:,1],columns=np.arange(X.shape[1]))
df_trimmed_2['label']=y[:,0]

# For label 0
print("-----------------------------------------------")
df_trimmed_1_part=df_trimmed_1.loc[df_trimmed_1['label']==2]
df_trimmed_1_part=df_trimmed_1_part.iloc[:,:-1]
# print(df_trimmed_1_part)

df_trimmed_2_part=df_trimmed_2.loc[df_trimmed_2['label']==2]
df_trimmed_2_part=df_trimmed_2_part.iloc[:,:-1]
# print(df_trimmed_2_part)

df_trimmed_1_part.replace(0, np.nan, inplace=True)
df_trimmed_2_part.replace(0, np.nan, inplace=True)

print(df_trimmed_1_part.isnull().sum())

print(df_trimmed_2_part.isnull().sum())


df_1_mean=df_trimmed_1_part.mean()
# print(df_1_mean)

df_2_mean=df_trimmed_2_part.mean()
# print(df_2_mean)

X_cleaned=[]
print(df_trimmed_1_part.shape)
for i in range(df_trimmed_1_part.shape[0]):
    l=[]
    for j in range(df_trimmed_1_part.shape[1]): 
      ele=(df_trimmed_1_part.iloc[i,j],df_trimmed_2_part.iloc[i,j])
      if((np.isnan(ele[0])==True) and (np.isnan(ele[1])==True)):
        # print("ele",ele)
        ele=(df_1_mean.iloc[j],df_2_mean.iloc[j])
      l.append([ele[0],ele[1]])
    l=np.array(l)
    X_cleaned.append(l)


# For label 1
print("--------------------------------------------------")
df_trimmed_1_part=df_trimmed_1.loc[df_trimmed_1['label']==3]
df_trimmed_1_part=df_trimmed_1_part.iloc[:,:-1]
# print(df_trimmed_1_part)

df_trimmed_2_part=df_trimmed_2.loc[df_trimmed_2['label']==3]
df_trimmed_2_part=df_trimmed_2_part.iloc[:,:-1]
# print(df_trimmed_2_part)

df_trimmed_1_part.replace(0, np.nan, inplace=True)
df_trimmed_2_part.replace(0, np.nan, inplace=True)

df_1_mean=df_trimmed_1_part.mean()
# print(df_1_mean)

df_2_mean=df_trimmed_2_part.mean()
# print(df_2_mean)

for i in range(df_trimmed_1_part.shape[0]):
    l=[]
    for j in range(df_trimmed_1_part.shape[1]): 
      ele=(df_trimmed_1_part.iloc[i,j],df_trimmed_2_part.iloc[i,j])
      if((np.isnan(ele[0])==True) and (np.isnan(ele[1])==True)):
        ele=(df_1_mean.iloc[j],df_2_mean.iloc[j])
      l.append([ele[0],ele[1]])
    l=np.array(l)
    X_cleaned.append(l)

X_cleaned=np.array(X_cleaned)
print(X_cleaned.shape)

print(X[0][0])
print(X_cleaned[0][0])

X_train,X_test,y_train,y_test=train_test_split(X_cleaned,y,test_size=0.3,shuffle=True)
# Ensure that X and Y are multiples of batch_size
if(X_train.shape[0]%64!=0):
    new_length=X_train.shape[0]-(X_train.shape[0]%64)
    X_train=X_train[:new_length,:,:]
    y_train=y_train[:new_length,:]
if(X_test.shape[0]%64!=0):
    new_length=X_test.shape[0]-(X_test.shape[0]%64)
    X_test=X_test[:new_length,:,:]
    y_test=y_test[:new_length,:]

print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

# Preparing a LSTM network for pose classification
import keras
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense
from keras.callbacks import CSVLogger


csv_logger=CSVLogger(filename='./training.log',append='True')

LSTM_model=Sequential()
LSTM_model.add(LSTM(10,input_shape=(25,2)))
LSTM_model.add(Dense(4,activation='softmax'))
print(LSTM_model.summary())

LSTM_model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
LSTM_model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=600,callbacks=[csv_logger])

LSTM_model.save('./model_full.h5')
LSTM_model.save_weights('./model_weights.h5')
json_model=LSTM_model.to_json()
with open("./model_arch.json","w") as fp:
  fp.write(json_model)

# VAE using TENSOR-FLOW (for 2 classes of yoga poses)
import importlib
import tensorflow as tf
import numpy as np
importlib.reload(tf)
tfd = tf.contrib.distributions
print(tf.VERSION)

original_dim=50
output_original_dim = 50
latent_dim = 8
intermediate_dim = 25
output_dim=50

X_train_flattened=X_train.reshape((X_train.shape[0],output_original_dim))
# X_test_flattened=X_test.reshape((X_test.shape[0],output_original_dim))

class History():
  def __init__(self):
    self.loss=[]
    self.val_loss=[]
    self.r_loss=[]
    self.KL_loss=[]
    self.mae_loss=[]
  def add(self,train_loss,r_loss,kl_loss):
    self.loss.append(train_loss)
    self.r_loss.append(r_loss)
    self.KL_loss.append(kl_loss)
    # self.loss.append(logs.get('loss'))
    # self.val_loss.append(logs.get('val_loss'))
    # self.reconstruction_loss.append(logs.get('binary_crossentropy'))
    # self.KL_loss.append(logs.get('loss')-(logs.get('binary_crossentropy')*30))
    # self.mae_loss.append(logs.get('custom_metric'))
recorded_history=History()

EPOCHS = 2000
BATCH_SIZE = 64

# using two numpy arrays
features, labels = (X_train_flattened, X_train_flattened)
dataset = tf.data.Dataset.from_tensor_slices((features,labels)).repeat().batch(BATCH_SIZE)
iterator = dataset.make_one_shot_iterator()
x, y = iterator.get_next()# make a simple model

z_log_var=0.0
z_mu=0.0
def make_encoder(data, code_size):
  global z_log_var,z_mu
  # x = tf.layers.flatten(data)
  e_1 = tf.layers.dense(data, intermediate_dim, tf.nn.relu)
  loc = tf.layers.dense(e_1, code_size) 
  log_var = tf.layers.dense(e_1, code_size) 
  scale = tf.exp(log_var/2)
  z_mu = loc
  z_log_var = log_var
  return tfd.MultivariateNormalDiag(loc, scale)

def make_prior(code_size):
  loc = tf.zeros(code_size, dtype=tf.dtypes.float64)
  scale = tf.ones(code_size, dtype=tf.dtypes.float64)
  return tfd.MultivariateNormalDiag(loc, scale)

def make_decoder(code, data_shape):
  s = code
  d_1 = tf.layers.dense(s, intermediate_dim, tf.nn.relu)
  d_2 = tf.layers.dense(d_1, output_dim , tf.nn.sigmoid)
  # logit = tf.layers.dense(z, np.prod(data_shape))
  # logit = tf.reshape(logit, [-1] + data_shape)
  # return tfd.Independent(tfd.Bernoulli(logit), 2)
  return d_2

make_encoder = tf.make_template('encoder', make_encoder)
make_decoder = tf.make_template('decoder', make_decoder)

# data = tf.placeholder(tf.float32, [None, original_dim])

prior = make_prior(code_size=latent_dim) # size of the latent space
posterior = make_encoder(x, code_size=latent_dim)
code = posterior.sample() 


predicted = make_decoder(code, [output_dim])
# Defining reconstruction loss

logit_predicted = tf.log(predicted/(1-predicted))
r_loss= 30 * -tf.reduce_mean((y * tf.log(1e-10 + predicted)) + ((1-y) * tf.log(1e-10 + 1 - predicted)),-1)
# Defining KL divergence

divergence = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mu) - tf.exp(z_log_var),-1)
# divergence = tfd.kl_divergence(posterior, prior)
# Defining ELBO loss function
total_loss = tf.reduce_mean(r_loss + divergence)

optimize = tf.train.AdamOptimizer(0.001).minimize(total_loss)
# samples = make_decoder(prior.sample(10), [25, 2]).mean()

# Define saver to save model
# saver = tf.train.Saver()
save_path = './'

with tf.Session() as sess2:
  # new_saver = tf.train.import_meta_graph(save_path + 'vae_tensorflow_model-2000.meta')
  # new_saver.restore(sess2, tf.train.latest_checkpoint(save_path))
  # print('model restored')
  sess2.run(tf.global_variables_initializer())
  sess2.run([optimize, total_loss, r_loss, divergence])
  graph = tf.get_default_graph()
  # w1 = graph.get_tensor_by_name("optimize:0")

  for epoch in range(EPOCHS):
    _, train_loss, train_r_loss,train_kl_loss = sess2.run(
        [optimize, total_loss, r_loss , divergence])
    print('Epoch', epoch, 'Loss', train_loss)

    print('-------------------------------')
    recorded_history.add(train_loss,train_r_loss,train_kl_loss)
  
  saver.save(sess2, save_path + 'vae_tensorflow_model')
  print('model saved')

# Variational autoencoders with flattened features.
# For 2 classes of yoga poses-
import keras
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense
from keras import metrics
from keras.callbacks import CSVLogger
import pandas as pd


# Hyperparameters
original_dim=50
output_original_dim = 50
batch_size=64
latent_dim = 8
intermediate_dim = 25
output_dim=50

# Network architecture

# csv_logger=CSVLogger(filename='./training.log',append='True')


class LossHistory(keras.callbacks.Callback):
  def on_train_begin(self,logs={}):
    self.loss=[]
    self.val_loss=[]
    self.reconstruction_loss=[]
    self.KL_loss=[]
    self.mae_loss=[]
  def on_epoch_end(self,epoch,logs={}):
    self.loss.append(logs.get('loss'))
    self.val_loss.append(logs.get('val_loss'))
    self.reconstruction_loss.append(logs.get('binary_crossentropy'))
    self.KL_loss.append(logs.get('loss')-(logs.get('binary_crossentropy')*30))
    self.mae_loss.append(logs.get('custom_metric'))

history=LossHistory()

X_train_flattened=X_train.reshape((X_train.shape[0],output_original_dim))
X_test_flattened=X_test.reshape((X_test.shape[0],output_original_dim))
print(X_train_flattened.shape)
print(X_test_flattened.shape)

def sampling(args):
  z_mean,z_log_var=args[0],args[1]
  epsilon=K.random_normal(shape=(64,8),mean=0.0)
  return epsilon*K.exp(z_log_var/2)+z_mean

# Define the network using Functional API
# Encoder
x=Input(batch_shape=(batch_size,original_dim),name="input_1")
dense_intermediate=Dense(intermediate_dim,activation="relu")
z_mean=Dense(latent_dim,name='Dense_mean')
z_log_var=Dense(latent_dim,name='Dense_log_var')

dense_intermediate_out=dense_intermediate(x)
z_mean_out=z_mean(dense_intermediate_out)
z_log_var_out=z_log_var(dense_intermediate_out)

# Lamda layer-Connecting layer
lat=Lambda(sampling)
z_out=lat([z_mean_out,z_log_var_out])

# Decoder
dense_2_layer=Dense(intermediate_dim,activation='relu')
output_layer=Dense(output_dim,activation='sigmoid')

dense_2_out=dense_2_layer(z_out)
output=output_layer(dense_2_out)

model=Model(x,output)


def custom_metric(y_true,y_pred):
  # Using MAE as a metric
  return metrics.mae(y_true,y_pred)

# Define vae Loss
# Takes parameters (y_true,y_pred)
def vae_loss(y_true,y_pred):
  reconstruct_loss=objectives.binary_crossentropy(y_true,y_pred)
  print(reconstruct_loss.shape) 
  # To introduce a constant to give more importance to the reconstruction loss
  reconstruct_loss=30 * reconstruct_loss
  kl_loss=-0.5*K.mean(z_log_var_out+1-K.square(z_mean_out)-K.exp(z_log_var_out),axis=-1)
  print(kl_loss.shape)
  # reconstruct_loss=K.print_tensor(reconstruct_loss,message='R loss= ')
  # history.resconstruction_loss.append(reconstruct_loss)
  return reconstruct_loss + kl_loss

def compile_model(model):
    model.compile(optimizer='Adam',loss=vae_loss,metrics=['binary_crossentropy',custom_metric])
    print(model.summary())

def train_model(model, epoch=2000):
    # Ensure that X_train and X_test are multiples of batch_size 
    model.fit(X_train_flattened,X_train_flattened,epochs=epoch,validation_data=(X_test_flattened,X_test_flattened),batch_size=64,callbacks=[history])

compile_model(model)
train_model(model, 2000)

def plot_graphs(history):
    # Plot loss and validation loss
    plt.figure(figsize=(15,5))
    plt.plot(np.arange(len(history.loss)),history.loss)
    plt.plot(np.arange(len(history.val_loss)),history.val_loss)
    plt.title("Loss over time")
    plt.legend(['Loss','Validation loss'])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    plt.figure()
    plt.plot(np.arange(len(history.reconstruction_loss)),history.reconstruction_loss)
    plt.title("Reconstruction loss over time")
    plt.xlabel("Epochs")
    plt.ylabel("Reconstruction loss")
    plt.show()

    plt.figure()
    plt.plot(np.arange(len(history.KL_loss)),history.KL_loss)
    plt.title("KL loss over time")
    plt.xlabel("Epochs")
    plt.ylabel("KL loss")
    plt.show()

    plt.figure()
    plt.plot(np.arange(len(history.mae_loss)),history.mae_loss)
    plt.title("MAE loss over time")
    plt.xlabel("Epochs")
    plt.ylabel("MAE loss")
    plt.show()
plot_graphs(history)

# Save history to a file
df_history=pd.DataFrame(data={'Loss':history.loss,'Val_loss':history.val_loss,'Reconstruction_loss':history.reconstruction_loss,'KL_loss':history.KL_loss,'MAE_loss':history.mae_loss})
df_history.to_csv("./history_logs.csv")

# Encoder
# del trained_encoder
# del trained_decoder

x=Input(batch_shape=(batch_size,original_dim),name="input_1")
intermediate_out=dense_intermediate(x)
z_mean_out=z_mean(intermediate_out)
z_log_var_out=z_log_var(intermediate_out)
z_out=lat([z_mean_out,z_log_var_out])

trained_encoder=Model(x,z_out)

# Decoder
encoded_input = Input(shape=(latent_dim,))
dense_2_out=dense_2_layer(encoded_input)
output_2=output_layer(dense_2_out)

trained_decoder = Model(encoded_input, output_2)

trained_encoder.save('./encoder_model.h5')

trained_decoder.save('./decoder_model.h5')

model.save_weights('./model_weights.h5')
model.save('./model_full.h5')
json_string = model.to_json()
with open('./model_arch.json','w') as json_file:
  json_file.write(json_string)


from keras.utils.vis_utils import plot_model

plot_model(model,to_file="./model_img.png",show_shapes=True,show_layer_names=True)

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from keras.models import load_model

# load weights
# del trained_encoder
# x=Input(batch_shape=(batch_size,original_dim))
# intermediate_out=dense_intermediate(x)
# z_mean_out=z_mean(intermediate_out)
# z_log_var_out=z_log_var(intermediate_out)
# z_out=lat([z_mean_out,z_log_var_out])

# trained_encoder=Model(x,z_out)
trained_encoder=load_model('./encoder_model.h5')

print(trained_encoder.summary())


encoded_list=[]
for i in range(X_train_flattened.shape[0]):
  # Plot PCA of latent space
  encoded_points=trained_encoder.predict(X_train_flattened[i*64:(i+1)*64,:],batch_size=64)
  encoded_list.extend(encoded_points)
print(len(encoded_list))
encoded_list=np.array(encoded_list)
print(encoded_list.shape)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(encoded_list)
# print(principalComponents)

plt.figure()
plt.scatter(principalComponents[:,0],principalComponents[:,1],c=y_train.flatten())
plt.colorbar()
plt.show()

print(pca.explained_variance_ratio_)

# T-SNE
poses_tsne = TSNE().fit_transform(encoded_list)
print(poses_tsne)
plt.figure()
plt.scatter(poses_tsne[:,0],poses_tsne[:,1],c=y_train.flatten())
plt.colorbar()
plt.show()

# Interpolation of the Latent space - 2 classes
import numpy as np
import random
from keras.models import load_model
import matplotlib.pyplot as plt


# global variables
latent_dim = 8
output_original_dim = 50
intermediate_dim = 25
output_dim = 50


# Model to get latent space

X_train_flattened=X_train.reshape((X_train.shape[0],output_original_dim))
X_test_flattened=X_test.reshape((X_test.shape[0],output_original_dim))
print(X_train_flattened.shape)
print(X_test_flattened.shape)

# Encoder
# del trained_encoder
# del trained_decoder
# x=Input(batch_shape=(batch_size,original_dim),name="input_1")
# intermediate_out=dense_intermediate(x)
# z_mean_out=z_mean(intermediate_out)
# z_log_var_out=z_log_var(intermediate_out)
# z_out=lat([z_mean_out,z_log_var_out])
# trained_encoder=Model(x,z_out)

# common_path = '/gdrive/My Drive/DL_Assignment/models/final_models/Cleaned_Data/method_10/'
# trained_encoder=load_model(common_path + 'encoder_model.h5')

trained_encoder=load_model('./encoder_model.h5')

# Decoder
# encoded_input = Input(shape=(latent_dim,))
# dense_2_out=dense_2_layer(encoded_input)
# output_2=output_layer(dense_2_out)
# trained_decoder = Model(encoded_input, output_2)
# trained_decoder=load_model(common_path + 'decoder_model.h5')

trained_decoder=load_model('./decoder_model.h5')

print(trained_decoder.summary())
# print(X_train.shape)
# print(X_train_mag.shape)
# print(y_train.shape)
# print(len(encoded_list))

# Interpolation along a line between the centroids of 2 classes
def linear_interpolation_line(encoder,decoder, data, labels, steps):
    n=steps+1

    encoded_list=[]
    for i in range(data.shape[0]):
      encoded_points=encoder.predict(data[i*64:(i+1)*64,:],batch_size=64)
      encoded_list.extend(encoded_points)
    print(encoded_list[0])
    print(labels[0])
    Z_a = np.array([encoded_list[i] for i in range(len(labels)) if labels[i][0]==2])
    Z_b = np.array([encoded_list[i] for i in range(len(labels)) if labels[i][0]==3])

    z_a_centroid = np.mean(Z_a, axis=0)
    z_b_centroid = np.mean(Z_b, axis=0)

    z_diff = z_a_centroid - z_b_centroid 
    
    inter=np.zeros((n,encoded_list[0].shape[0]))
    for i in range(n):
      inter[i] = z_b_centroid + (i/n)*z_diff

    return decoder.predict(inter)

def plot_pose_transition(predicted):
    # Plot predicted output
    classes={1:'Plank pose',0:'Mountain pose',3:'Tree pose',4:'Warrior pose'}
    print("Plank pose --> Mountain pose")
    predicted_reshaped=predicted.reshape(predicted.shape[0],25,2)
    print('Reshaped : ', predicted_reshaped.shape)
    for j in range(predicted.shape[0]):
        x_coor=predicted_reshaped[j][:,0]
        y_coor=predicted_reshaped[j][:,1]

        y_coor_negated=[-1*y_coor[i] for i in range(y_coor.shape[0])]
        y_coor_negated=np.array(y_coor_negated)
        # plt.scatter(x_coor,y_coor_negated)
        plt.scatter(x_coor[0],y_coor_negated[0], marker='x')
        plt.plot([x_coor[0], x_coor[1]], [y_coor_negated[0], y_coor_negated[1]]) # Nose to neck
        plt.plot([x_coor[1], x_coor[2]], [y_coor_negated[1], y_coor_negated[2]]) # Neck to Left arm
        plt.plot([x_coor[1], x_coor[5]], [y_coor_negated[1], y_coor_negated[5]]) # Neck to Righ arm
        plt.plot([x_coor[1], x_coor[8]], [y_coor_negated[1], y_coor_negated[8]]) # To pelvic

        plt.plot([x_coor[8], x_coor[11]], [y_coor_negated[8], y_coor_negated[11]]) # Pelvic to Left leg
        plt.plot([x_coor[8], x_coor[14]], [y_coor_negated[8], y_coor_negated[14]]) # Pelvic to right leg

        plt.plot([x_coor[11], x_coor[24]], [y_coor_negated[11], y_coor_negated[24]]) # Left foot to foot end
        plt.plot([x_coor[14], x_coor[21]], [y_coor_negated[14], y_coor_negated[21]]) # Right foot to foot end


        plt.title("Predicted # "+str(j))
        plt.show()

decoded_output = linear_interpolation_line(trained_encoder, trained_decoder,X_train_flattened, y_train, 10)
print('--------')
print(decoded_output.shape)
plot_pose_transition(decoded_output)
