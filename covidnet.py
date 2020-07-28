import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch import optim
import torchvision.models as models

from get_data import get_dataloader
from utils.utils import read_data_path, plot_confusion_matrix
from models.main import build_model

import numpy as np
from sklearn.metrics import precision_recall_fscore_support as prf
from sklearn.metrics import confusion_matrix


class CovidNet:
    def __init__(self, args):
        """Class for CovidNet model."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.model = build_model(args, self.device)


    def train(self, dataloader):
        """Trainer for the CovidNet.

        Args:
            dataloader (torch.data.DataLoader): dataloader iterator to be used to perform 
                                                the training.
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.model.train()

        for epoch in range(self.args.num_epochs):
            print('Epoch: {}'.format(epoch+1))
            total_loss = 0
            for imgs, label in dataloader:
                optimizer.zero_grad()

                outputs = self.model(imgs)
                loss = criterion(outputs, label)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print('Training... loss: {:.3}'.format(total_loss/len(dataloader)))

        torch.save(self.model.state_dict(), '{}/trained_parameters_{}.pth'.format(
                   self.args.model_path, self.args.model))


    def predict(self, dataloader):
        """Predict metrics.

        Args:
            dataloader (torch.data.DataLoader): dataloader iterator to be used to perform 
                                                mini-batch predictions.

        Returns:
            labels (list): List of real labels for each sample from the dataloader.
            predictions (list): List of predicted labels for each sample from the dataloader.
        """
        self.model.eval() #Setting the model to eval mode.
        predictions = []
        labels = []
        with torch.no_grad():
            for imgs, label in dataloader:
                imgs, label = imgs.float().to(self.device), label.long().to(self.device)

                outputs = self.model(imgs)
                _, pred = torch.max(outputs.data, 1)

                predictions.extend(pred.cpu().detach().numpy())
                labels.extend(label.cpu().detach().numpy())
                
        precision, recall, f1_score, _ = prf(labels, predictions, average='macro') 
        print("Precision : {:0.4f}, Recall : {:0.4f}, F1-score : {:0.4f}".format(
               precision, recall, f1_score))
        return labels, predictions

    
    def sample_images(self, 
                      list_of_labels, 
                      test_data_path='test_split_v3.txt', 
                      batch_size=64):
        """Sample images from the test set and wrap them into a dataloader.

        Args:
            list_of_labels (list): List of class labels to be used for the image sampling.
            test_data_path (str): Path of the file containing the metadata for test set.
            batch_size (int): size of the batch.

        Returns:
            img (torch.Tensor): torch.Tensor version of the preprocessed image.
            label (int): label of the image based on self.mapping.
    
        """
        possible_labels = ['COVID-19', 'pneumonia', 'normal']
        if len(list_of_labels)<1:
            raise ValueError('The list must contain at least one element.')
        elif len(np.setdiff1d(list_of_labels, possible_labels))>0:
            raise ValueError('All the elements must belong to one of the following classes: \
                  {}'.format(possible_labels))

        files = read_data_path(test_data_path)
        labels = np.array([file.split(' ')[2] for file in files])
        unique_labels, counts = np.unique(list_of_labels, return_counts=True)
        sampled_files = []
        for i, label in enumerate(unique_labels):
            ixs = np.where(labels==label)[0]
            selected_ixs = np.random.choice(ixs, counts[i], replace=False)
            sampled_files.extend([files[ix] for ix in selected_ixs])
        return get_dataloader(sampled_files)
        

    def plot_cm_matrix(self, labels, predictions, path='cm_matrix.png'):
        """Plot the confusion matrix for the data.

        Args:
            labels (list): List of real labels.
            predictions (list): List of predicted.
            path (str): path the confusion matrix plot to be saved.
        """
        cm_matrix = confusion_matrix(labels, predictions)
        plot_confusion_matrix(cm_matrix, ['Normal', 'Pneumonia', 'Covid-19'], 
                              self.args, normalize=True)