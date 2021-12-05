# -*- coding: utf-8 -*-

import pandas as pd
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from random import randint
import matplotlib.pyplot as plt

y_train = []
x_train = []
query_id = []
array_train_x1 = []
array_train_x0 = []


def extract_features(toks):
    # Get features
    features = []
    for tok in toks:
        features.append(float(tok.split(":")[1]))
    return features

def extract_query_data(tok):
     #Get queryid documentid
     query_features = [tok.split(":")[1]] #qid  
     return query_features

def get_format_data(data_path):
   with open(data_path, 'r', encoding='utf-8') as file:
         for line in file:
             data, _, comment = line.rstrip().partition("#") 
             toks = data.split()
             y_train.append(int(toks[0])) #relativity
             x_train.append(extract_features(toks[2:])) # doc features
             query_id.append(extract_query_data(toks[1])) #qid

#dd = get_format_data("../../embeddings/short_data.csv")           
#dd = get_format_data("test.txt")
        
def get_pair_doc_data(y_train, query_id):
    #Pair
    pairs = []
    tmp_x0 = []
    tmp_x1 = []
    for i in range(0, len(query_id) - 1):
        for j in range(i + 1, len(query_id)):
            #Documents under each query
            if query_id[i][0] != query_id[j][0]:
                break
            #Use document pairs with different relevance
            if (query_id[i][0] == query_id[j][0]) and (y_train[i] != y_train[j]):
                #Put the most relevant first, and keep the first doc in the document pair more relevant to the query than the second doc
                if y_train[i] > y_train[j]:
                    pairs.append([i,j])
                    tmp_x0.append(x_train[i])
                    tmp_x1.append(x_train[j])
                else:
                    pairs.append([j,i])
                    tmp_x0.append(x_train[j])
                    tmp_x1.append(x_train[i])
    #The corresponding subscript elements in array_train_x0 and array_train_x1 keep the previous element more relevant than the next element
    array_train_x0 = np.array(tmp_x0)
    array_train_x1 = np.array(tmp_x1)
    print('fond {} doc pairs'.format(len(pairs)))
    return len(pairs), array_train_x0, array_train_x1

#pair = get_pair_doc_data(y_train, query_id)

class Dataset(data.Dataset):
 
     def __init__(self, data_path):
         #  Parse the training data
         get_format_data(data_path)
         #  pair combination
         self.datasize, self.array_train_x0, self.array_train_x1 = get_pair_doc_data(y_train, query_id)
 
     def __getitem__(self, index):
         data1 = torch.from_numpy(self.array_train_x0[index]).float()
         data2 = torch.from_numpy(self.array_train_x1[index]).float()
         return data1, data2
 
     def __len__(self):
         return self.datasize
     
def get_loader(data_path, batch_size, shuffle, num_workers):
     dataset = Dataset(data_path)
     data_loader = torch.utils.data.DataLoader(
         dataset=dataset,
         batch_size = batch_size,
         shuffle = shuffle,
         num_workers=num_workers
     )
     return data_loader

#data_loader = get_loader("../../embeddings/short_data.csv", 100, False, 4)

class RankNet(nn.Module):
     def __init__(self, inputs, hidden_size, outputs):
         super(RankNet, self).__init__()
         self.model = nn.Sequential(
             nn.Linear(inputs, hidden_size),
             #nn.Dropout(0.5),
             nn.ReLU(inplace=True),
             #nn.LeakyReLU(0.2, inplace=True), #inplace is True, it will change the input data, otherwise the original input will not be changed, only new output will be generated
             nn.Linear(hidden_size, outputs),
             #nn.Sigmoid()
         )
         self.sigmoid = nn.Sigmoid()
 
     def forward(self, input_1, input_2):
         result_1 = self.model(input_1) #Predict input_1 score
         result_2 = self.model(input_2) #Predict input_2 score
         pred = self.sigmoid(result_1 - result_2) #input_1 is more relevant than input_2
         return pred
 
     def predict(self, input):
         result = self.model(input)
         return result
     
def train():
     #  Super parameters
     inputs = 50
     hidden_size = 5
     outputs = 1
     learning_rate = 0.2
     num_epochs = 10
     batch_size = 100
 
     model = RankNet(inputs, hidden_size, outputs)
     #Loss function and optimizer
     criterion = nn.BCELoss()
     optimizer = optim.Adadelta(model.parameters(), lr = learning_rate)
 
     base_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
     base_path = os.path.dirname(base_path)
     
     #Need to change it based on dataset location
     
     data_path = base_path + "/covidtweet_rr/data/libsvm/input_train.txt"
 
     data_loader = get_loader(data_path, batch_size, False, 4)
     total_step = len(data_loader)
     #    The batch size method is used here, not every time a pair of docs is passed in for forward and backward propagation
     #  (tips: There is also a way to input all docs pairs under each query as batches into the network for forward and backward, but Dataset and DataLoader cannot be used here)
     for epoch in range(num_epochs):
         for i, (data1, data2) in enumerate(data_loader):
             #print('Epoch [{}/{}], Step [{}/{}]'.format(epoch, num_epochs, i, total_step))
             data1 = data1
             data2 = data2
             label_size = data1.size()[0]
             pred = model(data1, data2)
             loss = criterion(pred, torch.from_numpy(np.ones(shape=(label_size, 1))).float())
             optimizer.zero_grad()
             loss.backward()
             optimizer.step()
         #if i % 10 == 0:
             print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
 
     torch.save(model.state_dict(), 'model.ckpt')

train()
     
def test():
     #test data
     base_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
     base_path = os.path.dirname(base_path)
     
     #Need to change it based on dataset location
     
     test_path = base_path + "/covidtweet_rr/data/libsvm/input_test.txt"
 
     #  Super parameters
     inputs = 50
     hidden_size = 5
     outputs = 1
     model = RankNet(inputs, hidden_size, outputs)
     model.load_state_dict(torch.load('model.ckpt'))
     
     with open(test_path, 'r', encoding='utf-8') as f:
          features = []
          for line in f:
              toks = line.split()
              feature = []
              for tok in toks[2:]:
                  _, value = tok.split(":")
                  feature.append(float(value))
              features.append(feature)
          features = np.array(features)
      #print(features)
          
     features = np.array(features)
     features = torch.from_numpy(features).float()
     predict_score = model.predict(features)
     
     return predict_score

result = test()
result = result.tolist()


plt.title("Ranknet")
plt.ylim(0,1)
plt.plot(result, color='blue')
