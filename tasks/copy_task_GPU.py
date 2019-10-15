# Copy task for DNC
import os
import torch
from torch import nn
from torch import optim

import numpy as np
import random

from DNC_GPU.dnc import DNC_Module              # Importing DNC implementation

class task_copy():

    def __init__(self):
        self.name = "copy_task_GPU"
        self.controller_size = 100 
        self.controller_layers = 1
        self.num_read_heads = 2
        self.num_write_heads = 1
        self.sequence_width = 8
        self.sequence_min_len = 5
        self.sequence_max_len = 20   # Default: 20
        self.memory_N = 128
        self.memory_M = 20
        self.num_batches = 10000
        self.batch_size = 1
        self.rmsprop_lr = 1e-4
        self.rmsprop_momentum = 0.9
        self.rmsprop_alpha = 0.95
        self.machine = None
        self.loss = None
        self.optimizer = None

    def get_task_name(self):
        return self.name

    def _data_maker(self, num_batches, batch_size, seq_width, min_len, max_len):    # Generates data for copy task
        # The input data vector will be of size (num_data_rows x batch_size x num_data_columns)
        #
        # 1 1 1 0 1  | 1 1 0 1 0 | 1 1 1 0 1 | 1 0 1 1 1
        # 0 0 1 0 1  | 0 1 0 1 1 | 0 1 0 0 1 | 0 0 1 1 0
        # 0 1 1 0 1  | 1 1 0 0 0 | 1 0 1 0 1 | 0 0 1 1 0
        #
        # Above is the example of data. num_data_rows = 3, num_data_columns = 5, batch_size = 4
        #
        # At a time we will give each row strip to the NTM for prediction as shown below. Therefore input size for one interaction will be (batch_size x num_data_columns)
        # 
        # 1 1 1 0 1  | 1 1 0 1 0 | 1 1 1 0 1 | 1 0 1 1 1
        
        for batch_num in range(num_batches+1):
            # All batches have the same sequence length
            seq_len = random.randint(min_len, max_len)
            seq = np.random.binomial(1, 0.5, (seq_len, batch_size, seq_width))  # Here, seq_len = num_data_rows and seq_width = num_data_columns
            seq = torch.from_numpy(seq)

            # The input includes an additional channel used for the delimiter
            inp = torch.zeros(seq_len + 1, batch_size, seq_width + 1)
            inp[:seq_len, :, :seq_width] = seq
            inp[seq_len, :, seq_width] = 1.0 # delimiter in our control channel
            outp = seq.clone()

            yield batch_num+1, inp.float(), outp.float()

    def init_dnc(self):
        self.machine = DNC_Module(self.sequence_width + 1, self.sequence_width, self.controller_size, self.controller_layers, self.num_read_heads, self.num_write_heads, self.memory_N, self.memory_M)
        self.machine.cuda()

    def init_loss(self):
        self.loss = nn.BCEWithLogitsLoss().cuda()  # Binary Cross Entropy Loss -> Sigmoid Activation + Cross Entropy Loss

    def init_optimizer(self):
        self.optimizer = optim.RMSprop(self.machine.parameters(), momentum = self.rmsprop_momentum, alpha = self.rmsprop_alpha, lr = self.rmsprop_lr)

    def calc_loss(self, Y_pred, Y):
        return self.loss(Y_pred, Y)

    def get_sample_data(self):  # Sample data for Testing
        batch_size = 1
        seq_len = random.randint(self.sequence_min_len, self.sequence_max_len)
        seq = np.random.binomial(1, 0.5, (seq_len, batch_size, self.sequence_width))  # Here, seq_len = num_data_rows and seq_width = num_data_columns
        seq = torch.from_numpy(seq)

        # The input includes an additional channel used for the delimiter
        inp = torch.zeros(seq_len + 1, batch_size, self.sequence_width + 1)
        inp[:seq_len, :, :self.sequence_width] = seq
        inp[seq_len, :, self.sequence_width] = 1.0 # delimiter in our control channel
        outp = seq.clone()
        return inp.float().cuda(), outp.float().cuda()
    
    def calc_cost(self, Y_out, Y, batch_size):
        y_out_binarized = torch.sigmoid(Y_out.cpu()).clone().data
        y_out_binarized.apply_(lambda x: 0 if x < 0.5 else 1)

        cost = torch.sum(torch.abs(y_out_binarized - Y.cpu().data))
        return cost.item()/batch_size

    def get_training_data(self):
        return self._data_maker(self.num_batches, self.batch_size, self.sequence_width, self.sequence_min_len, self.sequence_max_len)

    def test_model(self):
        self.machine.initialization(self.batch_size)    # Initializing states
        X, Y = self.get_sample_data()

        X, Y = X.cuda(), Y.cuda()
        Y_out = torch.zeros(Y.shape).cuda()

        # Feeding the DNC network all the data first and then predicting output
        # by giving zero vector as input and previous read states and hidden vector
        # and thus training vector this way to give outputs matching the labels

        for i in range(X.shape[0]):
            self.machine(X[i])

        for i in range(Y.shape[0]):
            Y_out[i, :, :], _ = self.machine()

        loss = self.calc_loss(Y_out, Y)
        cost = self.calc_cost(Y_out, Y, self.batch_size)    # The cost is the number of error bits per sequence

        print("\n\nTest Data - Loss: " + str(loss.cpu().item()) + ", Cost: " + str(cost))
        
        X.squeeze(1)
        Y.squeeze(1)
        Y_out = torch.sigmoid(Y_out.cpu().squeeze(1))

        print("\n------Input---------\n")
        print(X.cpu().data)
        print("\n------Labels---------\n")
        print(Y.cpu().data)
        print("\n------Output---------")
        print((Y_out.data).apply_(lambda x: 0 if x < 0.5 else 1))
        print("\n")

        return loss.cpu().item(), cost, X.cpu(), Y.cpu(), Y_out

    def clip_grads(self):       # Clipping gradients for stability
        """Gradient clipping to the range [10, 10]."""
        parameters = list(filter(lambda p: p.grad is not None, self.machine.parameters()))
        for p in parameters:
            p.grad.data.clamp_(-10, 10)

    def train_model(self):
        # Here, the model is optimized using BCE Loss, however, it is evaluated using Number of error bits in predction and actual labels (cost)
        loss_list = []
        cost_list = []
        seq_length = []
        
        if (self.num_batches/10) > 0:
            model_save_interval = (self.num_batches/10)
        else:
            model_save_interval = 1

        for batch_num, X, Y in self.get_training_data():

            if batch_num>self.num_batches:
                break

            self.optimizer.zero_grad()                      # Making old gradients zero before calculating the fresh ones
            self.machine.initialization(self.batch_size)    # Initializing states
            
            X, Y = X.cuda(), Y.cuda()

            Y_out = torch.zeros(Y.shape).cuda()

            # Feeding the NTM network all the data first and then predicting output
            # by giving zero vector as input and previous read states and hidden vector
            # and thus training vector this way to give outputs matching the labels

            for i in range(X.shape[0]):
                self.machine(X[i])

            for i in range(Y.shape[0]):
                Y_out[i, :, :], _ = self.machine()

            loss = self.calc_loss(Y_out, Y)
            loss.backward()
            self.clip_grads()
            self.optimizer.step()

            # The cost is the number of error bits per sequence
            cost = self.calc_cost(Y_out, Y, self.batch_size)

            loss_list += [loss.item()]
            cost_list += [cost]
            seq_length += [Y.shape[0]]

            if batch_num % model_save_interval == 0:
                self.save_model(batch_num)

            print("Batch: " + str(batch_num) + "/" + str(self.num_batches) + ", Loss: " + str(loss.item()) + ", Cost: " + str(cost) + ", Sequence Length: " + str(Y.shape[0]))

    def save_model(self, curr_epoch):
        if not os.path.exists("Saved_Models/" + self.name):
            os.mkdir("Saved_Models/" + self.name)
        state_dic = {'task_name': self.name, 'start_epoch': curr_epoch + 1, 'state_dict': self.machine.state_dict(), 'optimizer_dic' : self.optimizer.state_dict()}
        filename = "Saved_Models/" + self.name + "/" + self.name + "_" + str(curr_epoch) + "_saved_model.pth.tar"
        torch.save(state_dic, filename)

    def load_model(self, option, epoch):
        path = "Saved_Models/" + self.name + "/" + self.name + "_" + str(epoch) + "_saved_model.pth.tar"
        if option == 1:             # Loading for training
            checkpoint = torch.load(path)
            self.machine.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_dic'])
        else:                       # Loading for testing
            checkpoint = torch.load(path)
            self.machine.load_state_dict(checkpoint['state_dict'])
            self.machine.eval()