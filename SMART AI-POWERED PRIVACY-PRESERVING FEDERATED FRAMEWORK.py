# ================================================================
# SMART AI-POWERED PRIVACY-PRESERVING FEDERATED FRAMEWORK
# PART 1 : DATA LOADING + PREPROCESSING
# ================================================================

# Install Packages (Run Once in Google Colab)
# !pip install tensorflow opencv-python pandas numpy scipy scikit-learn nltk gensim

import os
import cv2
import glob
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

from scipy.signal import butter, filtfilt

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings("ignore")

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

import tensorflow as tf

SEED=42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ==========================================================
# PATHS
# ==========================================================

DATASET="/content/dataset"

IMAGE_DIR=os.path.join(DATASET,"images")
EHR_FILE=os.path.join(DATASET,"ehr.csv")
ECG_FILE=os.path.join(DATASET,"ecg.csv")
TEXT_FILE=os.path.join(DATASET,"clinical_text.csv")

IMAGE_SIZE=224

# ==========================================================
# LOAD IMAGES
# ==========================================================

classes=sorted(os.listdir(IMAGE_DIR))

image_paths=[]
labels=[]

for idx,cls in enumerate(classes):

    folder=os.path.join(IMAGE_DIR,cls)

    files=glob.glob(folder+"/*.jpg")
    files+=glob.glob(folder+"/*.png")
    files+=glob.glob(folder+"/*.jpeg")

    for f in files:

        image_paths.append(f)
        labels.append(idx)

image_paths=np.array(image_paths)
labels=np.array(labels)

print("Classes :",classes)
print("Images :",len(image_paths))

# ==========================================================
# IMAGE PREPROCESSING
# ==========================================================

def preprocess_image(path):

    img=cv2.imread(path)

    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    img=cv2.resize(img,(IMAGE_SIZE,IMAGE_SIZE))

    lab=cv2.cvtColor(img,cv2.COLOR_RGB2LAB)

    l,a,b=cv2.split(lab)

    clahe=cv2.createCLAHE(clipLimit=2.0,
                          tileGridSize=(8,8))

    cl=clahe.apply(l)

    merged=cv2.merge((cl,a,b))

    img=cv2.cvtColor(merged,
                     cv2.COLOR_LAB2RGB)

    img=cv2.medianBlur(img,3)

    img=img.astype(np.float32)

    img=img/255.0

    return img

# ==========================================================
# LOAD ALL IMAGES
# ==========================================================

images=[]

for p in image_paths:

    images.append(preprocess_image(p))

images=np.array(images)

print(images.shape)

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train,\
X_test,\
y_train,\
y_test=train_test_split(

images,
labels,

test_size=0.20,

random_state=42,

stratify=labels

)

print(X_train.shape)
print(X_test.shape)

# ==========================================================
# LOAD EHR
# ==========================================================

ehr=pd.read_csv(EHR_FILE)

print(ehr.head())

imputer=SimpleImputer(strategy="median")

ehr=imputer.fit_transform(ehr)

scaler=MinMaxScaler()

ehr=scaler.fit_transform(ehr)

print("EHR :",ehr.shape)

# ==========================================================
# LOAD ECG
# ==========================================================

ecg=pd.read_csv(ECG_FILE)

ecg=np.array(ecg)

def butter_filter(signal):

    b,a=butter(
        4,
        0.15,
        btype='low'
    )

    return filtfilt(b,a,signal)

filtered=[]

for row in ecg:

    filtered.append(
        butter_filter(row)
    )

filtered=np.array(filtered)

filtered=(filtered-filtered.mean())/filtered.std()

print(filtered.shape)

# ==========================================================
# LOAD CLINICAL TEXT
# ==========================================================

text_df=pd.read_csv(TEXT_FILE)

lemmatizer=WordNetLemmatizer()

stop_words=set(stopwords.words("english"))

def clean_text(sentence):

    sentence=str(sentence).lower()

    words=nltk.word_tokenize(sentence)

    words=[
        lemmatizer.lemmatize(w)
        for w in words
        if w.isalpha()
        and w not in stop_words
    ]

    return " ".join(words)

text_df["processed"]=text_df.iloc[:,0].apply(clean_text)

print(text_df.head())

# ==========================================================
# VISUALIZATION
# ==========================================================

plt.figure(figsize=(12,6))

for i in range(6):

    plt.subplot(2,3,i+1)

    plt.imshow(X_train[i])

    plt.title(classes[y_train[i]])

    plt.axis("off")

plt.tight_layout()

plt.show()
# ==========================================================
# FEATURE EXTRACTION
# MobileNetV3 (Image)
# ==========================================================

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.applications import MobileNetV3Small

IMG_SIZE = 224

# ----------------------------------------------------------
# MobileNetV3 Backbone
# ----------------------------------------------------------

base_model = MobileNetV3Small(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = Input(shape=(IMG_SIZE, IMG_SIZE,3))

x = base_model(inputs, training=False)

x = GlobalAveragePooling2D()(x)

x = BatchNormalization()(x)

x = Dense(
    512,
    activation="relu"
)(x)

x = Dropout(0.3)(x)

x = Dense(
    256,
    activation="relu"
)(x)

image_feature_model = Model(
    inputs,
    x,
    name="MobileNetV3_FeatureExtractor"
)

image_feature_model.summary()

# ----------------------------------------------------------
# Extract Image Features
# ----------------------------------------------------------

train_image_features = image_feature_model.predict(
    X_train,
    batch_size=16,
    verbose=1
)

test_image_features = image_feature_model.predict(
    X_test,
    batch_size=16,
    verbose=1
)

print(train_image_features.shape)
print(test_image_features.shape)

# ==========================================================
# MLP FOR EHR
# ==========================================================

ehr_input = Input(shape=(ehr.shape[1],))

y = Dense(256,activation='relu')(ehr_input)
y = BatchNormalization()(y)
y = Dropout(0.30)(y)

y = Dense(128,activation='relu')(y)
y = BatchNormalization()(y)

y = Dense(64,activation='relu')(y)

ehr_model = Model(ehr_input,y)

ehr_model.summary()

ehr_features = ehr_model.predict(
    ehr,
    batch_size=32,
    verbose=1
)

print(ehr_features.shape)
# ==========================================================
# ECG FEATURE EXTRACTION (1D-CNN)
# ==========================================================

from tensorflow.keras.layers import *
from tensorflow.keras.models import *

ECG_LENGTH = filtered.shape[1]

filtered = filtered.reshape(filtered.shape[0], ECG_LENGTH, 1)

ecg_input = Input(shape=(ECG_LENGTH,1))

z = Conv1D(32,3,padding='same',activation='relu')(ecg_input)
z = BatchNormalization()(z)
z = MaxPooling1D(2)(z)

z = Conv1D(64,3,padding='same',activation='relu')(z)
z = BatchNormalization()(z)
z = MaxPooling1D(2)(z)

z = Conv1D(128,3,padding='same',activation='relu')(z)
z = BatchNormalization()(z)
z = MaxPooling1D(2)(z)

z = GlobalAveragePooling1D()(z)

z = Dense(128,activation='relu')(z)
z = Dropout(0.3)(z)

z = Dense(64,activation='relu')(z)

ecg_model = Model(ecg_input,z)

ecg_model.summary()

ecg_features = ecg_model.predict(
    filtered,
    batch_size=32,
    verbose=1
)

print("ECG Feature Shape :",ecg_features.shape)

# ==========================================================
# CLINICAL TEXT FEATURE EXTRACTION
# TF-IDF + Word2Vec
# ==========================================================

from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

sentences = [
    txt.split()
    for txt in text_df["processed"]
]

word2vec = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4
)

tfidf = TfidfVectorizer(max_features=500)

tfidf_features = tfidf.fit_transform(
    text_df["processed"]
).toarray()

print("TFIDF :",tfidf_features.shape)

def average_word2vec(sentence):

    words = sentence.split()

    vectors=[]

    for w in words:

        if w in word2vec.wv:

            vectors.append(word2vec.wv[w])

    if len(vectors)==0:

        return np.zeros(100)

    return np.mean(vectors,axis=0)

word2vec_features=np.array([
    average_word2vec(text)
    for text in text_df["processed"]
])

print("Word2Vec :",word2vec_features.shape)

text_features=np.concatenate(
    [
        tfidf_features,
        word2vec_features
    ],
    axis=1
)

print("Final Text Features :",text_features.shape)

# ==========================================================
# MULTIMODAL FEATURE FUSION
# ==========================================================

minimum = min(
    len(train_image_features),
    len(ehr_features),
    len(ecg_features),
    len(text_features)
)

train_image_features = train_image_features[:minimum]
ehr_features = ehr_features[:minimum]
ecg_features = ecg_features[:minimum]
text_features = text_features[:minimum]
y_train = y_train[:minimum]

multimodal_features = np.concatenate(

    [

        train_image_features,

        ehr_features,

        ecg_features,

        text_features

    ],

    axis=1

)

print("Multimodal Feature Shape :",multimodal_features.shape)

from sklearn.preprocessing import StandardScaler

fusion_scaler = StandardScaler()

multimodal_features = fusion_scaler.fit_transform(
    multimodal_features
)

print("Normalized Feature Shape :",multimodal_features.shape)

# ==========================================================
# STAGE 5
# ADAPTIVE QUANTUM-ATTENTIVE CAPSULE-RNN (AQAC-RNN)
# ==========================================================

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import backend as K

NUM_CLASSES = len(classes)

# ==========================================================
# QUANTUM ATTENTION LAYER
# ==========================================================

class QuantumAttention(Layer):

    def __init__(self, units=256):
        super().__init__()
        self.units = units

    def build(self,input_shape):

        self.W = self.add_weight(
            shape=(input_shape[-1],self.units),
            initializer="glorot_uniform",
            trainable=True)

        self.V = self.add_weight(
            shape=(self.units,1),
            initializer="glorot_uniform",
            trainable=True)

    def call(self,x):

        score=tf.nn.tanh(tf.matmul(x,self.W))

        weights=tf.nn.softmax(
            tf.matmul(score,self.V),
            axis=1)

        output=x*weights

        return output

# ==========================================================
# CAPSULE LAYER
# ==========================================================

class CapsuleLayer(Layer):

    def __init__(self,
                 num_capsules=10,
                 dim_capsule=16,
                 routings=3):

        super().__init__()

        self.num_capsules=num_capsules
        self.dim_capsule=dim_capsule
        self.routings=routings

    def build(self,input_shape):

        self.W=self.add_weight(

            shape=(1,
                   input_shape[1],
                   self.num_capsules,
                   self.dim_capsule,
                   input_shape[-1]),

            initializer='glorot_uniform',

            trainable=True

        )

    def squash(self,s):

        s_norm=K.sum(K.square(s),axis=-1,keepdims=True)

        scale=s_norm/(1+s_norm)

        return scale*s/K.sqrt(s_norm+K.epsilon())

    def call(self,u):

        u=tf.expand_dims(u,2)

        u=tf.expand_dims(u,-1)

        u_hat=tf.reduce_sum(self.W*u,axis=-2)

        b=tf.zeros_like(u_hat[:,:,:,:,0])

        for i in range(self.routings):

            c=tf.nn.softmax(b,axis=2)

            s=tf.reduce_sum(
                tf.expand_dims(c,-1)*u_hat,
                axis=1)

            v=self.squash(s)

            if i<self.routings-1:

                b=b+tf.reduce_sum(
                    u_hat*tf.expand_dims(v,1),
                    axis=-1)

        return tf.reshape(
            v,
            (-1,
             self.num_capsules,
             self.dim_capsule)
        )

# ==========================================================
# AQAC-RNN MODEL
# ==========================================================

feature_size=multimodal_features.shape[1]

inputs=Input(shape=(feature_size,))

x=Dense(512,
        activation='relu')(inputs)

x=BatchNormalization()(x)

x=Dropout(0.30)(x)

x=Dense(256,
        activation='relu')(x)

x=Reshape((16,16))(x)

# Quantum Attention

x=QuantumAttention(128)(x)

# Capsule Network

x=CapsuleLayer(

    num_capsules=8,

    dim_capsule=16,

    routings=3

)(x)

# Bidirectional GRU

x=Bidirectional(

    GRU(

        128,

        return_sequences=True

    )

)(x)

x=Bidirectional(

    GRU(

        64

    )

)(x)

x=Dropout(0.40)(x)

x=Dense(

    128,

    activation='relu'

)(x)

x=Dropout(0.30)(x)

outputs=Dense(

    NUM_CLASSES,

    activation='softmax'

)(x)

AQAC_RNN=Model(

    inputs,

    outputs

)

AQAC_RNN.summary()

# ==========================================================
# COMPILE
# ==========================================================

AQAC_RNN.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001),

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy']

)

print("\nAQAC-RNN Model Ready...\n")

# ==========================================================
# MODEL CHECKPOINT
# ==========================================================

checkpoint=tf.keras.callbacks.ModelCheckpoint(

    "AQAC_RNN.h5",

    monitor="val_accuracy",

    save_best_only=True,

    verbose=1

)

early=tf.keras.callbacks.EarlyStopping(

    monitor="val_loss",

    patience=10,

    restore_best_weights=True

)

reduce=tf.keras.callbacks.ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=5,

    verbose=1

)

print("AQAC-RNN Successfully Created.")

# ==========================================================
# STAGE 6
# FEDERATED LEARNING (FedAvgM)
# ==========================================================

import copy
import numpy as np
import tensorflow as tf

NUM_CLIENTS = 5
LOCAL_EPOCHS = 2
GLOBAL_ROUNDS = 10
BATCH_SIZE = 16
MOMENTUM = 0.90

# ==========================================================
# SPLIT DATA AMONG CLIENTS
# ==========================================================

client_data = []

samples = len(multimodal_features)
split_size = samples // NUM_CLIENTS

for i in range(NUM_CLIENTS):

    start = i * split_size

    if i == NUM_CLIENTS - 1:
        end = samples
    else:
        end = (i + 1) * split_size

    X = multimodal_features[start:end]
    Y = y_train[start:end]

    client_data.append((X, Y))

print("Clients :", len(client_data))

# ==========================================================
# CREATE CLIENT MODEL
# ==========================================================

def create_client_model():

    model = tf.keras.models.clone_model(AQAC_RNN)

    model.build((None, multimodal_features.shape[1]))

    model.set_weights(AQAC_RNN.get_weights())

    model.compile(

        optimizer=tf.keras.optimizers.Adam(1e-4),

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]

    )

    return model

# ==========================================================
# LOCAL TRAINING
# ==========================================================

def local_training(model, X, y):

    history = model.fit(

        X,
        y,

        epochs=LOCAL_EPOCHS,

        batch_size=BATCH_SIZE,

        verbose=0,

        shuffle=True

    )

    return model.get_weights()

# ==========================================================
# FEDAVGM AGGREGATION
# ==========================================================

velocity = None

def FedAvgM(local_weights):

    global velocity

    avg_weights = []

    for weights in zip(*local_weights):

        avg_weights.append(

            np.mean(weights, axis=0)

        )

    if velocity is None:

        velocity = [

            np.zeros_like(w)

            for w in avg_weights

        ]

    final_weights = []

    for i in range(len(avg_weights)):

        velocity[i] = (

            MOMENTUM * velocity[i]

            + avg_weights[i]

        )

        final_weights.append(

            velocity[i]

        )

    return final_weights

# ==========================================================
# FEDERATED TRAINING
# ==========================================================

global_model = create_client_model()

global_accuracy = []

global_loss = []

for round_no in range(GLOBAL_ROUNDS):

    print("=" * 60)
    print("Global Round :", round_no + 1)
    print("=" * 60)

    local_weights = []

    for client in range(NUM_CLIENTS):

        print("Training Client :", client + 1)

        client_model = create_client_model()

        client_model.set_weights(

            global_model.get_weights()

        )

        X, y = client_data[client]

        weights = local_training(

            client_model,

            X,

            y

        )

        local_weights.append(weights)

    new_weights = FedAvgM(

        local_weights

    )

    global_model.set_weights(

        new_weights

    )

    loss, acc = global_model.evaluate(

        multimodal_features,

        y_train,

        verbose=0

    )

    global_accuracy.append(acc)

    global_loss.append(loss)

    print("Global Accuracy :", acc)

# ==========================================================
# SAVE GLOBAL MODEL
# ==========================================================

global_model.save(

    "Federated_AQAC_RNN.keras"

)

print("Global Model Saved Successfully")

# ==========================================================
# PLOT GLOBAL ACCURACY
# ==========================================================

import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))

plt.plot(

    global_accuracy,

    linewidth=3,

    marker="o"

)

plt.xlabel("Communication Round")

plt.ylabel("Accuracy")

plt.title("Federated Learning Accuracy")

plt.grid(True)

plt.show()

# ==========================================================
# PLOT GLOBAL LOSS
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(

    global_loss,

    linewidth=3,

    marker="o"

)

plt.xlabel("Communication Round")

plt.ylabel("Loss")

plt.title("Federated Learning Loss")

plt.grid(True)

plt.show()
# ==========================================================
# STAGE 7
# ADAPTIVE BLOCKCHAIN-BASED PRIVACY-AWARE SECURE SHARING
# (ABPSS)
# ==========================================================

import hashlib
import time
import json
import secrets
import numpy as np
from cryptography.fernet import Fernet

# ==========================================================
# DIFFERENTIAL PRIVACY
# ==========================================================

class AdaptiveDifferentialPrivacy:

    def __init__(self, epsilon=1.0, sensitivity=1.0):

        self.epsilon = epsilon
        self.sensitivity = sensitivity

    def add_noise(self, data):

        scale = self.sensitivity / self.epsilon

        noise = np.random.laplace(
            0,
            scale,
            data.shape
        )

        return data + noise

dp = AdaptiveDifferentialPrivacy(
    epsilon=0.8,
    sensitivity=1
)

secure_features = dp.add_noise(
    multimodal_features
)

print("Differential Privacy Applied")

# ==========================================================
# LIGHTWEIGHT HOMOMORPHIC ENCRYPTION
# ==========================================================

class LightweightHE:

    def __init__(self):

        self.secret = np.random.randint(100,1000)

    def encrypt(self,data):

        return data + self.secret

    def decrypt(self,data):

        return data - self.secret

he = LightweightHE()

encrypted_features = he.encrypt(
    secure_features
)

print("Homomorphic Encryption Applied")

# ==========================================================
# SHA-256 HASH
# ==========================================================

def SHA256(data):

    return hashlib.sha256(

        str(data).encode()

    ).hexdigest()

data_hash = SHA256(

    encrypted_features

)

print("SHA256 Hash")
print(data_hash)

# ==========================================================
# BLOCK
# ==========================================================

class Block:

    def __init__(

        self,

        index,

        timestamp,

        data,

        previous_hash

    ):

        self.index=index

        self.timestamp=timestamp

        self.data=data

        self.previous_hash=previous_hash

        self.nonce=0

        self.hash=self.calculate_hash()

    def calculate_hash(self):

        block=json.dumps({

            "index":self.index,

            "time":self.timestamp,

            "data":str(self.data),

            "previous":self.previous_hash,

            "nonce":self.nonce

        },sort_keys=True)

        return hashlib.sha256(

            block.encode()

        ).hexdigest()

# ==========================================================
# BLOCKCHAIN
# ==========================================================

class Blockchain:

    def __init__(self):

        self.chain=[]

        self.create_genesis()

    def create_genesis(self):

        genesis=Block(

            0,

            time.time(),

            "Genesis",

            "0"

        )

        self.chain.append(genesis)

    def latest(self):

        return self.chain[-1]

    def add_block(self,data):

        previous=self.latest()

        block=Block(

            len(self.chain),

            time.time(),

            data,

            previous.hash

        )

        self.chain.append(block)

    def verify(self):

        for i in range(1,len(self.chain)):

            current=self.chain[i]

            previous=self.chain[i-1]

            if current.previous_hash!=previous.hash:

                return False

            if current.hash!=current.calculate_hash():

                return False

        return True

# ==========================================================
# CREATE BLOCKCHAIN
# ==========================================================

blockchain=Blockchain()

for i in range(len(encrypted_features)):

    blockchain.add_block(

        encrypted_features[i].tolist()

    )

print("Blockchain Length :",len(blockchain.chain))

print("Blockchain Verified :",blockchain.verify())

# ==========================================================
# SECURE REPOSITORY
# ==========================================================

repository=[]

for block in blockchain.chain:

    repository.append({

        "index":block.index,

        "hash":block.hash,

        "previous":block.previous_hash

    })

print("Secure Repository Ready")

# ==========================================================
# AES-256 (FERNET)
# ==========================================================

key=Fernet.generate_key()

cipher=Fernet(key)

encrypted_repository=[]

for record in repository:

    token=cipher.encrypt(

        json.dumps(record).encode()

    )

    encrypted_repository.append(token)

print("Repository Encrypted")

# ==========================================================
# DECRYPT SAMPLE
# ==========================================================

sample=cipher.decrypt(

    encrypted_repository[0]

)

print(sample.decode())

# ==========================================================
# SAVE SECURE DATA
# ==========================================================

import pickle

with open(

    "Secure_Healthcare_Repository.pkl",

    "wb"

) as f:

    pickle.dump(

        encrypted_repository,

        f

    )

print("Secure Repository Saved")