#!/usr/bin/env python
# coding: utf-8

# In[80]:


bucket = 'sagemaker-1026'
prefix = 'sagemaker/DEMO-linear-mnist'
key = 'stratified_train.txt'
 
# Define IAM role
import boto3
import re
from sagemaker import get_execution_role

role = get_execution_role()
s3 = boto3.client('s3')


# In[81]:


import io
import numpy as np
import sagemaker.amazon.common as smac
import csv

train = []
data = s3.get_object(Bucket=bucket, Key=key)
data = data['Body'].read().decode('utf-8')
datas = data.split('\n')

for i in range(len(datas)):
    datas[i] = datas[i].split('\t')
    for j in range(len(datas[i][j])):
        datas[i][j] = int(datas[i][j])
    train.append(datas[i])
    

train= np.array(train, dtype='float32')
train = np.transpose(train)

print('train', train, train.shape)

#train= np.loadtxt('stratified_train.txt', unpack=True, dtype='float32')
#print('train', train, train.shape)

vectors = np.transpose(train[0:50]) #shape (1256, 50)
labels = np.transpose(train[50:]) #(1256, 1)
print ("labels", labels.shape)
labels=np.concatenate(labels)  #(1256, )


print(len(labels.shape))


buf = io.BytesIO()
smac.write_numpy_to_dense_tensor(buf, vectors, labels)
buf.seek(0)


# In[75]:


import boto3
import os

key = 'recordio-pb-data'
boto3.resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train', key)).upload_fileobj(buf)
s3_train_data = 's3://{}/{}/train/{}'.format(bucket, prefix, key)
print('uploaded training data location: {}'.format(s3_train_data))


# In[76]:


output_location = 's3://{}/{}/output'.format(bucket, prefix)
print('training artifacts will be uploaded to: {}'.format(output_location))


# In[77]:



from sagemaker.amazon.amazon_estimator import get_image_uri
container = get_image_uri(boto3.Session().region_name, 'linear-learner')


# In[ ]:


import boto3
import sagemaker

sess = sagemaker.Session()

linear = sagemaker.estimator.Estimator(container,
                                       role, 
                                       train_instance_count=1, 
                                       train_instance_type='ml.c4.xlarge',
                                       output_path=output_location,
                                       sagemaker_session=sess)
linear.set_hyperparameters(feature_dim=50,
                           predictor_type='multiclass_classifier',
                           num_classes=96,
                           mini_batch_size=200,
                           epochs=72,
                           accuracy_top_k=3)

linear.fit({'train': s3_train_data})


# In[7]:


linear_predictor = linear.deploy(initial_instance_count=1,
                                 instance_type='ml.m4.xlarge')


# In[8]:


from sagemaker.predictor import csv_serializer, json_deserializer

linear_predictor.content_type = 'text/csv'
linear_predictor.serializer = csv_serializer
linear_predictor.deserializer = json_deserializer


# In[9]:


test_list=[]
test_label_list=[]

file = open('stratified_test.csv', 'r',encoding='UTF-8') 
rdr = csv.reader(file)
for line in rdr:
    test_list.append(line)

file1 = open('stratified_test_label.csv', 'r',encoding='UTF-8') 
rdr1 = csv.reader(file1)
for line in rdr1:
    test_label_list.append(line)

for i in range(len(test_list)):
    for j in range(len(test_list[i])):
        a=int(test_list[i][j])
        test_list[i][j]=a

for i in range(len(test_label_list)):
    for j in range(len(test_label_list[i])):
        #print(type(train_label_list[i][j]))
        b=int(test_label_list[i][j])
        test_label_list[i][j]=b
        
test_vectors = np.array(test_list).astype('float32')
test_labels = np.array(test_label_list).astype('float32')

test_labels=np.concatenate(test_labels)


# In[10]:


classes = []
file = open('class_name.txt', 'r',encoding='UTF-8') 
rdr = csv.reader(file)
for line in rdr:
    classes.append(line)
        
total=0
for i in range(len(test_list)):
    result = linear_predictor.predict(test_list[i])
    result1=result['predictions'][0]['score']
    idx = np.argsort(result1)[-3:]
    num=test_labels[i]
    print('original: ',classes[int(num)])
    for j in range(3):
        print(classes[idx[j]])
        if(idx[j]==num):
            total=total+1
            break
print(total/len(test_list))


# In[ ]:





# In[ ]:




