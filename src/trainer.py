import os, copy, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import dataset, Subset
from random import random, sample
from time import time
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
import matplotlib.pyplot as plt
from model import *


class Trainer():
    def __init__(self, training_configs, model_configs, data_configs) -> None:
        self.num_epochs = training_configs["num_epochs"]  # Number of full passes through the dataset
        self.batch_size = training_configs["batch_size"]  # Number of samples in each minibatch
        self.learning_rate = training_configs["learning_rate"]
        self.seed = training_configs["seed"]  # Seed the random number generator for reproducibility
        self.device = training_configs['device'] if torch.cuda.is_available() else 'cpu'
        print(f"Using device {self.device}")
        
        self.fig_dir = training_configs['fig_dir']
        self.log_dir = training_configs['log_dir']
        self.model_dir = training_configs['model_dir']
        self.overwrite_model = training_configs['overwrite_model']
        self.save_as_script = training_configs['save_as_script']
        self.model_configs = model_configs
        self.data_configs = data_configs
        self.model = Desnow(model_configs=self.model_configs, data_configs=self.data_configs).to(self.device).half()
        self.criterion = CustomLoss(loss_configs=self.model_configs['loss_configs'])
        self.optimizer = training_configs['optimizer'](self.model.parameters(), lr=self.learning_rate,
                                                        weight_decay=training_configs['l2_reg'])
        self.model_name = model_configs['model_name']

    
    def __param_to_device(self, inputs, trues, masks):
        inputs = inputs.to(self.device)
        trues = trues.to(self.device)
        masks = masks.to(self.device)
        return inputs, trues, masks
    

    def train_epoch(self, model, optimizer, train_loader):
        model.train()
            
        train_loss = 0
        for _, (inputs, trues, masks) in enumerate(train_loader):
            optimizer.zero_grad()
            print(inputs.shape)
            inputs, y, z = self.__param_to_device(inputs, trues, masks)
            y_snow_free, y_hat, z_hat = model(inputs)
            # print(outputs.shape)
            # print(outputs[0])
            loss = self.criterion(y, z, y_snow_free, y_hat, z_hat)
            print("backpropping")
            loss.backward()
            optimizer.step()
            train_loss += loss.item()/trues.shape[0]

        
        train_loss = train_loss/len(train_loader)
        return train_loss
    

    def val_epoch(self, model, val_loader):
        model.eval()
            
        val_loss = 0
        val_acc = 0
        last_batch_len = 0
        last_batch_acc = 0
        with torch.no_grad():
            for _, (inputs, trues, masks) in enumerate(val_loader):
                inputs, y, z = self.__param_to_device(inputs, trues, masks)
                y_snow_free, y_hat, z_hat = model(inputs).squeeze(1)
                #print(outputs.shape)
                loss = self.criterion(y, z, y_snow_free, y_hat, z_hat)
                val_loss += loss.item()/trues.shape[0]
                
        val_acc = val_acc/len(val_loader)
        
        # print('LAST BATCH LENGTH:', last_batch_len)
        # print('LAST BATCH ACC:', last_batch_acc)
        return val_loss
    

    def __pick_best_model(self, model, best_model, val_loss, best_loss):
        if val_loss < best_loss:
            best_model = copy.deepcopy(model)
            best_loss = val_loss
            # torch.save(best_model.state_dict(), os.path.join(self.model_dir, self.data_type+'_'+self.model))
        return best_model, best_loss
    

    def train(self, training_dataloader, val_dataloader):
        start = time()
        best_loss, val_loss = 1e99, 1e99
        best_model = None
        self.history = {'train_loss':[], 'val_loss':[]}
        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch(self.model, self.optimizer, training_dataloader)
            val_loss = self.val_epoch(self.model, val_dataloader)
            best_model, best_loss = self.__pick_best_model(self.model, best_model, val_loss, best_loss)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            print(f"Epoch:{epoch+1}/{self.num_epochs}")
            print(f"AVG Training Loss: {train_loss:.4f}, AVG Val Loss: {val_loss:.4f}")

        end = time()
        print(f"training for {self.num_epochs} epochs took {end-start:2}s")
        path = os.path.join(self.model_dir, f'{self.model_name}.pt')
        if not os.path.exists(path) or self.overwrite_model:
            if self.save_as_script:
                best_model_script = torch.jit.trace(best_model,
                    torch.rand(1, 3, self.data_configs['sample_shape']).to(self.device))
                torch.jit.save(best_model_script, path)
            else:
                torch.save(best_model.state_dict(), path)
        else:
            print(f'model {self.model_name}.pt already exists, not overwriting')
            
        return best_model


    def plot_history(self):
        x = np.arange(self.num_epochs)
        fig = plt.figure()
        plt.title(f'{self.model_name} Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(x+1, self.history['train_loss'], label='training loss')
        plt.plot(x+1, self.history['val_loss'], label='val loss')
        plt.legend()
        path = os.path.join(self.fig_dir, f'{self.model_name}_history.png')
        if not os.path.exists(path) or self.overwrite_model:
            plt.savefig(os.path.join(self.fig_dir, f'{self.model_name}_history.png'))
        else:
            print(f'figure {self.model_name}_history.png already exists, not overwriting')


    def train_val_test(self, training_dataloader, val_dataloader, test_l_dataloader, test_m_dataloader, test_s_dataloader):
        # with open(os.path.join(self.log_dir, self.model_name+'_out.txt'), 'w') as sys.stdout, \
        #     open(os.path.join(self.log_dir, self.model_name+'_err.txt'), 'w') as sys.stderr:
        
        print("begun training")
        best_model = self.train(training_dataloader, val_dataloader)
        print("done training")
        del training_dataloader, val_dataloader

        test_l_loss = self.val_epoch(best_model, test_l_dataloader)
        test_m_loss = self.val_epoch(best_model, test_m_dataloader)
        test_s_loss = self.val_epoch(best_model, test_s_dataloader)
        del test_l_dataloader, test_m_dataloader, test_s_dataloader
        
        print(f"Test Loss L: {test_l_loss:.4f}")
        print(f"Test Loss M: {test_m_loss:.4f}")
        print(f"Test Loss S: {test_s_loss:.4f}")
        return best_model, self.history, test_l_loss, test_m_loss, test_s_loss



        


        




