#!/usr/bin/env python
# coding: utf-8

# In[80]:


#library import

import pandas as pd
import librosa 
import librosa.display
import IPython.display as ipd 
import numpy as np
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
#Read the csv file

audio_dataset_path = 'C:/Users/rkyada/OneDrive/Desktop/coughwavcovid/'
metadata_path = pd.read_csv('C:/Users/rkyada/OneDrive/Desktop/coughwavcovid/cough_covid.csv') 


#Show data
metadata_path.head(30) 


# In[81]:


#we extract 100 characteristics from each sound file
def audio_feature_extraction(file):
    audio, sample_rate = librosa.load(audio_dataset_path+file_name, res_type='kaiser_fast')
    mfcc_feat = librosa.feature.mfcc(y=audio, sr= sample_rate, n_mfcc=100)
    mfcc_scaled_features = np.mean(mfcc_feat.T, axis =0)
    return mfcc_scaled_features

import numpy as np
extracted_features = []
for index_num, row in tqdm(metadata_path.iterrows()):
    file_name = str(row['file_name'])
    final_class_labels = str(row['class'])
    data = audio_feature_extraction(file_name)
    extracted_features.append([data, final_class_labels])


# In[106]:


### converting extracted_features to Pandas dataframe
extracted_features_df=pd.DataFrame(extracted_features,columns=['feature','class'])
extracted_features_df.tail(10)



# In[83]:


# Split the dataset into independent and dependent dataset
X=np.array(extracted_features_df['feature'].tolist())
y=np.array(extracted_features_df['class'].tolist())
X.shape
y


# In[84]:


#Select training and test data from the model
labelencoder=LabelEncoder()
y=to_categorical(labelencoder.fit_transform(y))
### Train Test Split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[85]:


#Show the data
print("Training data:")
print(X_train.shape)
print(y_train.shape)

print("Test data:")
print(X_test.shape)
print(y_test.shape)


# In[86]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense,Activation, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
## No of classes
num_labels=y.shape[1]
print(num_labels)


# In[87]:


#Building the layers
model=Sequential()
###first layer
model.add(Dense(100,input_shape=(100,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###second layer
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###third layer
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))

###final layer
model.add(Dense(num_labels))
model.add(Activation('softmax'))
model.summary()


# In[88]:


#Compile the models
model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')


# In[97]:


## Training 
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime 

num_epochs = 400
num_batch_size = 10
#save the training
checkpointer = ModelCheckpoint(filepath='cought_covid.h5', 
                               verbose=1, save_best_only=True)
start = datetime.now()
print("Start Training...")
history=model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=1)


duration = datetime.now() - start
print("Training completed in time: ", duration)


# In[90]:


#Show the grahp accuray
test_accuracy=model.evaluate(X_test,y_test,verbose=0)
print(test_accuracy[1])


import matplotlib.pyplot as plt
plt.figure(figsize=[14,10])
plt.subplot(211)
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
 
# Accuracy Curves
plt.figure(figsize=[14,10])
plt.subplot(212)
plt.plot(history.history['accuracy'],'r',linewidth=3.0)
plt.plot(history.history['val_accuracy'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)


# The model shows a 100% success rate.
# It should be noted that the amount of training dataset is very small.
# 
# Let's try new files after training ... 

# In[91]:


# function to extract features from the audion file

# transform each category with it's respected label

def extract_feature(file_name):
    # load the audio file
    audio_data, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    
    # get the feature 
    feature = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=100)
    
    # scale the features
    feature_scaled = np.mean(feature.T,axis=0)
    
    # return the array of features
    return np.array([feature_scaled])




# function to predict the feature

def print_prediction(file_name):
    
    # extract feature from the function defined above
    prediction_feature = extract_feature(file_name) 
    
    # get the id of label using argmax
    predicted_vector = np.argmax(model.predict(prediction_feature), axis=-1)
    
    # get the class label from class id
    predicted_class = labelencoder.inverse_transform(predicted_vector)
    
    # display the result
    print("The predicted class is:", predicted_class[0], '\n')
        


# In[107]:


# File name=file to predict
#upload files that have not been used in the model

file_name ='C://Users//rkyada//Downloads//sample-38.wav'

# get the output
print_prediction(file_name)

# play the file

dat1, sampling_rate1 = librosa.load(file_name)
plt.figure(figsize=(20, 10))
D = librosa.amplitude_to_db(np.abs(librosa.stft(dat1)), ref=np.max)
plt.subplot(4, 2, 1)
librosa.display.specshow(D, y_axis='linear')
librosa
plt.colorbar(format='%+2.0f dB')
plt.title(file_name)

ipd.Audio(file_name)


# In[ ]:





# In[ ]:




