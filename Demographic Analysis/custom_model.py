import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage.feature import local_binary_pattern
from sklearn.preprocessing import StandardScaler, LabelEncoder

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Add, Dense, MaxPooling2D, Conv2D, Input, Activation, GlobalAveragePooling2D, Concatenate
from tensorflow.keras.regularizers import l2

import warnings
warnings.filterwarnings("ignore")

### Dataset Setup ###
AGE_MAP = {
    '0-2': [0],
    '3-9': [1],
    '10-19': [2],
    '20-29': [3],
    '30-39': [4],
    '40-49': [5],
    '50-59': [6],
    '60-69': [7],
    'more than 70': [8]
}
def age_mapper(age):
  return AGE_MAP[age]

le_gender = LabelEncoder()
le_race = LabelEncoder()

### TRAINING DATA PREPARATION ###
train_df = pd.read_csv('fairface_label_train.csv')
train_gen_feat_df = pd.read_csv('gender_features_train.csv')
train_gen_feat_df = train_gen_feat_df[train_gen_feat_df['file'].duplicated(keep=False) == False]
filenames = train_gen_feat_df['file'].values
train_df = train_df[train_df['file'].isin(filenames)]

train_gender = np.array(le_gender.fit_transform(train_df[['gender']]), dtype=np.int32).reshape((-1,1))
train_race = np.array(le_race.fit_transform(train_df[['race']]), dtype=np.int32).reshape((-1,1))
train_race = to_categorical(train_race, num_classes=7)
train_age = np.array(list(map(age_mapper, train_df['age'].values.tolist())))
train_labels = {"output_gender": train_gender, "output_race": train_race, "output_age": train_age}

### VALIDATION DATA PREPARATION ###
val_df = pd.read_csv('fairface_label_val.csv')
val_gen_feat_df = pd.read_csv('gender_features_val.csv')
val_gen_feat_df = val_gen_feat_df[val_gen_feat_df['file'].duplicated(keep=False) == False]
filenames = val_gen_feat_df['file'].values
val_df = val_df[val_df['file'].isin(filenames)]

val_gender = np.array(le_gender.fit_transform(val_df[['gender']]), dtype=np.int32).reshape((-1,1))
val_race = np.array(le_race.fit_transform(val_df[['race']]), dtype=np.int32).reshape((-1,1))
val_race = to_categorical(val_race, num_classes=7)
val_age = np.array(list(map(age_mapper, val_df['age'].values.tolist())))
val_labels = {"output_gender": val_gender, "output_race": val_race, "output_age": val_age}

### APPENDING EXTRACTED GEOMETRIC FACIAL FEATURES ###
scaler = StandardScaler()
intermediate_train_features = scaler.fit_transform(train_gen_feat_df.drop('file', axis=1))
intermediate_val_features = scaler.fit_transform(val_gen_feat_df.drop(['gender', 'file'], axis=1))

train_fns = train_df['file'].values
train_inputs = { 'input1': train_fns, 'input2': intermediate_train_features}
train_ds = tf.data.Dataset.from_tensor_slices((train_inputs, train_labels))

val_fns = val_df['file'].values
val_inputs = { 'input1': val_fns, 'input2': intermediate_val_features}
val_ds = tf.data.Dataset.from_tensor_slices((val_inputs, val_labels))

# strategy = tf.distribute.MirroredStrategy()
# train_dist_ds = strategy.experimental_distribute_dataset(train_ds)
# val_dist_ds = strategy.experimental_distribute_dataset(val_ds)

def lbp_compute(img, radius, num_points):
  b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
  lbp_b = local_binary_pattern(b, num_points, radius, method='uniform')
  lbp_g = local_binary_pattern(g, num_points, radius, method='uniform')
  lbp_r = local_binary_pattern(r, num_points, radius, method='uniform')
  lbp = np.concatenate([lbp_b.ravel(), lbp_g.ravel(), lbp_r.ravel()])
  lbp = lbp.reshape((224, 224, -1))
  return lbp

# Map the filenames to their corresponding images
def load_and_preprocess_image(inputs, labels):
    img = tf.io.read_file(inputs['input1'])
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.py_function(func=lbp_compute, inp=[img, 8, 1], Tout=tf.float32)
    inputs['input1'] = img
    return inputs, labels

batch_size = 16

train_ds = train_ds.map(load_and_preprocess_image)
train_ds = train_ds.batch(batch_size)
val_ds = val_ds.map(load_and_preprocess_image)
val_ds = val_ds.batch(batch_size)


### MODEL ###
# with strategy.scope():
def Convolution(input_tensor,filters):
    # x = Conv2D(filters=filters,kernel_size=(3, 3),padding = 'same',strides=(1, 1),kernel_regularizer=l2(0.01))(input_tensor)
    x = Conv2D(filters=filters, kernel_size=(3,3), padding='same', strides=(1,1))(input_tensor)
    x = Dropout(0.15)(x)
    x = Activation('relu')(x)
    return x

def model(img_input_shape, param_input_shape):
    inputs = Input(shape=(img_input_shape), name='input1')

    conv_1 = Convolution(inputs, 64)
    maxp_1 = MaxPooling2D(pool_size = (2,2)) (conv_1)
    conv_2 = Convolution(maxp_1, 128)
    maxp_2 = MaxPooling2D(pool_size = (2,2)) (conv_2)
    conv_3 = Convolution(maxp_2, 256)
    maxp_3 = MaxPooling2D(pool_size = (2,2)) (conv_3)
    conv_4 = Convolution(maxp_3, 512)
    maxp_4 = MaxPooling2D(pool_size = (2,2)) (conv_4)

    global_pool = GlobalAveragePooling2D()(maxp_4)

    dlib_inputs = Input(shape=(param_input_shape), name='input2')
    combined = Concatenate()([global_pool, dlib_inputs])

    dense_11 = Dense(32, activation='relu')(combined)
    dense_1 = Dense(64,activation='relu')(dense_11)
    dense_2 = Dense(128,activation='relu')(dense_1)
    dense_3 = Dense(256,activation='relu')(dense_2)
    dense_4 = Dense(512,activation='relu')(dense_3)

    # reshape1 = Dense(512,activation='relu')(combined)
    reshape1 = Conv2D(filters=505, activation='relu', kernel_size=(1,1), padding='same')(maxp_4)
    reshape1 = GlobalAveragePooling2D()(reshape1)
    reshape1 = Concatenate()([reshape1, dlib_inputs])
    skip1 = Add()([reshape1, dense_4])
    hidden2 = Dense(128, activation='relu')(skip1)

    hidden3 = Dense(256,activation='relu')(hidden2)
    dense_5 = Dense(256,activation='relu')(hidden3)
    dense_6 = Dense(128,activation='relu')(dense_5)
    hidden4 = Dense(128,activation='relu')(dense_6)

    # reshape2 = Dense(128,activation='relu')(combined)
    reshape2 = Conv2D(filters=121, activation='relu', kernel_size=(1,1), padding='same')(maxp_4)
    reshape2 = GlobalAveragePooling2D()(reshape2)
    reshape2 = Concatenate()([reshape2, dlib_inputs])
    skip2 = Add()([reshape2, hidden4])
    dense_7= Dense(128,activation='relu')(skip2)

    dense_8 = Dense(64, activation='relu')(dense_7)
    dense_9 = Dense(32, activation='relu')(dense_8)

    output_gender = Dense(1,activation="sigmoid",name='output_gender')(dense_9)
    output_age = Dense(1,activation="linear",name='output_age')(dense_9)
    output_race = Dense(7, activation="softmax", name="output_race")(dense_9)

    model = Model(inputs=[inputs, dlib_inputs], outputs=[output_gender, output_age, output_race])
    model.compile(
        loss=['binary_crossentropy', 'mean_squared_error', 'categorical_crossentropy'], 
        optimizer="adam", 
        metrics={'output_gender': 'accuracy', 'output_age': 'mae', 'output_race': 'accuracy'})
    
    return model

### TRAINING ###

img_shape = (224, 224, 3)
param_shape = (7)
custom_model = model(img_shape, param_shape)
custom_model.summary()

epochs = 40
# with strategy.scope():
learnt = custom_model.fit(train_ds, epochs=epochs, validation_data=val_ds)

# test_image, _ = load_and_preprocess_image('female.jpeg', '')
# test_image = np.expand_dims(test_image, axis=0)
# print("######## PREDICTION #######")
# pred = custom_model.predict(test_image)

custom_model.save('8model')

model_history_string = str(learnt.history)
with open('8model_history.txt', 'w') as f:
   f.write(model_history_string)


### Plots ###
gender_train_loss = learnt.history['output_gender_loss']
gender_val_loss = learnt.history['val_output_gender_loss']
gender_train_acc = learnt.history['output_gender_accuracy']
gender_val_acc = learnt.history['val_output_gender_accuracy']

age_train_loss = learnt.history['output_age_loss']
age_val_loss = learnt.history['val_output_age_loss']
age_train_mae = learnt.history['output_age_mae']
age_val_mae = learnt.history['val_output_age_mae']

race_train_loss = learnt.history['output_race_loss']
race_val_loss = learnt.history['val_output_race_loss']
race_train_acc = learnt.history['output_race_accuracy']
race_val_acc = learnt.history['val_output_race_accuracy']

# Loss Plots
# plt.figure(figsize=(10, 10))
# plt.title('Gender Loss Plot')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.plot(gender_train_loss, label="Training")
# plt.plot(gender_val_loss, label="Validation")
# plt.legend()
# plt.savefig('3gender_loss.png')

# plt.figure(figsize=(10, 10))
# plt.title('Age Loss Plot')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.plot(age_train_loss, label="Training")
# plt.plot(age_val_loss, label="Validation")
# plt.legend()
# plt.savefig('3age_loss.png')

# plt.figure(figsize=(10, 10))
# plt.title('Race Loss Plot')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.plot(race_train_loss, label="Training")
# plt.plot(race_val_loss, label="Validation")
# plt.legend()
# plt.savefig('3race_loss.png')

plt.figure(figsize=(10, 10))
plt.title('Loss Plot')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(gender_train_loss, label="Gender Training")
plt.plot(gender_val_loss, label="Gender Validation")
plt.plot(age_train_loss, label="Age Training")
plt.plot(age_val_loss, label="Age Validation")
plt.plot(race_train_loss, label="Race Training")
plt.plot(race_val_loss, label="Race Validation")
plt.legend()
plt.savefig('8loss.png')

# Accuracy Plots
# plt.figure(figsize=(10, 10))
# plt.title('Gender Accuracy Plot')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.plot(gender_train_acc, label="Training")
# plt.plot(gender_val_acc, label="Validation")
# plt.legend()
# plt.savefig('3gender_acc.png')

# plt.figure(figsize=(10, 10))
# plt.title('Age MAE Plot')
# plt.xlabel('Epoch')
# plt.ylabel('MAE')
# plt.plot(age_train_mae, label="Training")
# plt.plot(age_val_mae, label="Validation")
# plt.legend()
# plt.savefig('3age_acc.png')

# plt.figure(figsize=(10, 10))
# plt.title('Race Accuracy Plot')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.plot(race_train_acc, label="Training")
# plt.plot(race_val_acc, label="Validation")
# plt.legend()
# plt.savefig('3race_acc.png')

plt.figure(figsize=(10, 10))
plt.title('Accuracy/MAE Plot')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(gender_train_acc, label="Gender Training")
plt.plot(gender_val_acc, label="Gender Validation")
plt.plot(age_train_mae, label="Age Training (MAE)")
plt.plot(age_val_mae, label="Age Validation (MAE)")
plt.plot(race_train_acc, label="Race Training")
plt.plot(race_val_acc, label="Race Validation")
plt.legend()
plt.savefig('8acc.png')
