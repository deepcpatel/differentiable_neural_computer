# bAbI question answering task for DNC
# Note: In this task, some functions for data processing are adapted from the Github User bgavran's implementation of DNC on Github
import os
import re
import torch
import pickle
from torch import nn
from torch import optim
import torch.nn.functional as F

import numpy as np
import random

from DNC_GPU.dnc import DNC_Module  # Importing DNC Implementation

class task_babi():

    def __init__(self):
        self.name = "bAbI_task_GPU"
        self.controller_size = 128
        self.controller_layers = 1
        self.num_read_heads = 1
        self.num_write_heads = 1
        self.sequence_width = -1    # Length of each one hot vector of word (both input and output)
        self.sequence_len = -1      # Word length of each story
        self.memory_N = 128
        self.memory_M = 128
        self.num_batches = -1
        self.num_epoch = 1
        self.batch_size = 10
        self.adam_lr = 1e-4
        self.adam_betas = (0.9, 0.999)
        self.adam_eps = 1e-8
        self.machine = None
        self.loss = None
        self.optimizer = None
        self.ind_to_word = None
        self.data_dir = "Data/bAbI/en-10k"   # Data directory

    def get_task_name(self):
        return self.name

    def init_dnc(self):
        if not os.path.isfile("Data/sequence_width.txt"):
            self.read_data()    # To set the sequence width
        else:
            self.sequence_width = pickle.load(open("Data/sequence_width.txt",'rb'))  # To set the sequence width

        self.machine = DNC_Module(self.sequence_width, self.sequence_width, self.controller_size, self.controller_layers, self.num_read_heads, self.num_write_heads, self.memory_N, self.memory_M)
        self.machine.cuda()     # Enabling GPU

    def init_loss(self):
        self.loss = nn.CrossEntropyLoss(reduction = 'none').cuda()  # Cross Entropy Loss in Pytorch -> Softmax Activation + Cross Entropy Loss

    def init_optimizer(self):
        self.optimizer = optim.Adam(self.machine.parameters(), lr = self.adam_lr, betas = self.adam_betas, eps = self.adam_eps)

    def calc_loss(self, Y_pred, Y, mask):
        # Y: dim -> (sequence_len x batch_size)
        # Y_pred: dim -> (sequence_len x batch_size x sequence_width)
        # mask: dim -> (sequence_len x batch_size)

        loss_vec = torch.empty(Y.shape, dtype=torch.float32).cuda()

        for i in range(Y_pred.shape[0]):
            loss_vec[i, :] = self.loss(Y_pred[i], Y[i])

        return torch.sum(loss_vec*mask)/torch.sum(mask)

    def calc_cost(self, Y_pred, Y, mask):       # Calculates % Cost
        # Y: dim -> (sequence_len x batch_size)
        # Y_pred: dim -> (sequence_len x batch_size x sequence_width)
        # mask: dim -> (sequence_len x batch_size)
        Y_pred, Y, mask = Y_pred.cpu(), Y.cpu(), mask.cpu()
        return torch.sum(((F.softmax(Y_pred, dim=2).max(2)[1]) == Y).type(torch.long)*mask.type(torch.long)).item(), torch.sum(mask).item()

    def print_word(self, word_vec):         # Prints the word from word vector
        # "word_vect" dimension : (1 x sequence_width)
        idx = np.argmax(word_vec, axis = 1)
        word = self.ind_to_word[idx]
        print(word + "\n")

    def to_one_hot(self, story):            # Converts a vector into one hot form
        out_token = []

        I = np.eye(self.sequence_width)
        for idx in story:
            out_token.append(I[int(idx)])
        
        if len(out_token)>self.sequence_len:
            self.sequence_len = len(out_token)
        return out_token

    def padding_labels(self, stories):  # Making separate funcion to pad labels because, labels will not be in one-hot vector form due to the requirements of PyTorch Cross Entropy Loss Function
        padded_stories = []

        for story in stories:
            if len(story)<self.sequence_len:
                li = [1 for i in range(self.sequence_len - len(story))]
                story.extend(li)
            padded_stories.append(np.asarray(story, dtype = np.long))
        return padded_stories

    def padding(self, stories):  # Pads padding element to make all the stories of equal length
        padded_stories = []

        for story in stories:
            if len(story)<self.sequence_len:
                li = self.to_one_hot(np.ones(self.sequence_len - len(story)))
                story.extend(li)
            padded_stories.append(np.asarray(story, dtype = np.float32))
        return padded_stories

    def flatten_if_list(self, l):                   # Merges all the list within a list with the outer list elements. Example: [you', '?', ['-']] -> ['you', '?', '-']
        newl = []
        for elem in l:
            if isinstance(elem, list):              # Checking whether the element is 'list' or not
                newl.extend(elem)                   # input.extend(li_2) method appends all the elements of 'li_2' list into 'input' list
            else:
                newl.append(elem)
        return newl

    def structure_data(self, x, y):                                     # Prepares data for bAbI task
        # Preparing  Data
        keys = list(x.keys())
        random.shuffle(keys)                                            # Randomly Shuffling the key list

        inp_story = []
        out_story = []

        for key in keys:
            inp_story.extend(x[key])
            out_story.extend(y[key])

        story_idx = list(range(0, len(inp_story)))
        random.shuffle(story_idx)

        # Here I am breaking stories into different files because A single list can't store all the stories
        num_batch = int(len(story_idx)/self.batch_size)
        self.num_batches = num_batch
        counter = 1

        # Out Data
        x_out = []
        y_out = []
        mask_inp = []   # Will be used for making the mask to make "non amswer" output words from DNC irrelevent

        for i in story_idx:
            if num_batch <= 0:
                break

            x_out.append(self.to_one_hot(inp_story[i]))
            y_out.append(out_story[i])
            mask_inp.append(inp_story[i])       # Appending input story For making the mask

            if counter % self.batch_size == 0:
                # Resetting Counter
                counter = 0
                
                # Padding
                x_out_array = torch.tensor(np.asarray(self.padding(x_out)).swapaxes(0, 1))                              # Converting from (batch_size x story_length x word size) to (story_length x batch_size x word size)
                y_out_array = torch.tensor(np.asarray(self.padding_labels(y_out)).swapaxes(0, 1), dtype=torch.long)     # Converting from (batch_size x story_length x 1) to (story_length x batch_size x 1)
                m_inp_array = torch.tensor(np.asarray(self.padding_labels(mask_inp)).swapaxes(0, 1), dtype=torch.long)  # Converting from (batch_size x story_length x 1) to (story_length x batch_size x 1)

                # Renewing List and updating batch number
                x_out = []
                y_out = []
                mask_inp = []
                num_batch -= 1

                yield (self.num_batches - num_batch), x_out_array, y_out_array, (m_inp_array == 0).float()
            counter += 1

    def read_data(self):              # Reading and Cleaning data from the file
        storage_file = "Data/cleaned_data_bAbI_" + self.data_dir.split('/')[2] +".txt"

        if not os.path.isfile(storage_file):
            output_symbol = "-"                 # Indicates an expectation of output to the DNC
            newstory_delimiter = " NEWSTORY "   # To separate stories
            pad_symbol = "*"                    # Padding symbol

            file_paths = []

            word_to_ind = {output_symbol: 0, pad_symbol: 1}     # Dictionary to store indices of all the word in the bAbI dataset. Predefined symbols already stored
            all_input_stories = {}
            all_output_stories = {}

            # Making list of all the files in the data directory
            for f in os.listdir(self.data_dir):
                f_path = os.path.join(self.data_dir, f)
                if os.path.isfile(f_path):
                    file_paths.append(f_path)

            # Processing the text files
            for file_path in file_paths:
                # print(file_path)
                # Cleaning the text
                file = open(file_path).read().lower()
                file = re.sub("\n1 ", newstory_delimiter, file)     # Adding a delimeter between two stories
                file = re.sub(r"\d+|\n|\t", " ", file)              # Removing all numbers, newlines and tabs
                file = re.sub("([?.])", r" \1", file)               # Adding a space before all punctuations
                stories = file.split(newstory_delimiter)            # Splitting whole text into the stories

                input_stories = []          # Stores the stories into the index form, where each word has unique index
                output_stories = []

                # Tokenizing the text
                for i, story in enumerate(stories):
                    input_tokens = story.split()                                # Input stories are meant for inputting to the DNC
                    output_tokens = story.split()                               # Output stories works as labels

                    for i, token in enumerate(input_tokens):                    # This when encountered "?", replaces answers with "-" sign in the input for the SNC
                        if token == "?":
                            output_tokens[i + 1] = output_tokens[i + 1].split(",")
                            input_tokens[i + 1] = [output_symbol for _ in range(len(output_tokens[i + 1]))]

                    input_tokens = self.flatten_if_list(input_tokens)
                    output_tokens = self.flatten_if_list(output_tokens)

                    # Calculating index of all the words
                    for token in output_tokens:
                        if token not in word_to_ind:
                            word_to_ind[token] = len(word_to_ind)   

                    input_stories.append([word_to_ind[elem] for elem in input_tokens])      # Storing each story into a list of word indices form
                    output_stories.append([word_to_ind[elem] for elem in output_tokens])

                all_input_stories[file_path] = input_stories                                # Storing all the stories for each file
                all_output_stories[file_path] = output_stories

            # Dumping all the cleaned data into a file
            pickle.dump((word_to_ind, all_input_stories, all_output_stories),open(storage_file,'wb'))
            pickle.dump(len(word_to_ind),open("Data/sequence_width.txt",'wb'))
            self.sequence_width = len(word_to_ind)      # Vector length of one hot vector
        else:
            word_to_ind, all_input_stories, all_output_stories = pickle.load(open(storage_file,'rb'))
        return word_to_ind, all_input_stories, all_output_stories

    def get_training_data(self):                                                        # Data directory
        word_to_ind, all_input_stories, all_output_stories = self.read_data()
        self.ind_to_word = {ind: word for word, ind in word_to_ind.items()}             # Reverse Index to Word dictionary to show final output

        # Separating Test and Train Data
        x_train_stories = {k: v for k, v in all_input_stories.items() if k[-9:] == "train.txt"}
        y_train_stories = {k: v for k, v in all_output_stories.items() if k[-9:] == "train.txt"}
        return self.structure_data(x_train_stories, y_train_stories)      # dim: x_train, y_train -> A list of (sequence_len x sequence_width) sized stories

    def get_test_data(self):  # Sample data for Testing                                                    # Data directory
        _, all_input_stories, all_output_stories = self.read_data()

        # Separating Test and Train Data
        x_test_stories = {k: v for k, v in all_input_stories.items() if k[-8:] == "test.txt"}
        y_test_stories = {k: v for k, v in all_output_stories.items() if k[-8:] == "test.txt"}
        return self.structure_data(x_test_stories, y_test_stories)        # dim: x_test, y_test -> A list of (sequence_len x sequence_width) sized stories

    def test_model(self):   # Testing the model
        correct = 0
        total = 0
        print("\n")

        for batch_num, X, Y, mask in self.get_test_data():
            self.machine.initialization(self.batch_size)    # Initializing states
            Y_out = torch.zeros(X.shape)

            # Feeding the DNC network all the data first and then predicting output
            # by giving zero vector as input and previous read states and hidden vector
            # and thus training vector this way to give outputs matching the labels

            X, Y, mask = X.cuda(), Y.cuda(), mask.cuda()       # Sending to CUDA device

            for i in range(X.shape[0]):
                Y_out[i, :, :], _ = self.machine(X[i])

            corr, tot = self.calc_cost(Y_out, Y, mask)

            correct += corr
            total += tot
            print("Test Example " + str(batch_num) + "/" + str(self.num_batches) + " processed, Batch Accuracy: " + str((float(corr)/float(tot))*100.0) + " %")
        
        accuracy = (float(correct)/float(total))*100.0
        print("\nOverall Accuracy: " + str(accuracy) + " %")
        return accuracy         # in %

    def clip_grads(self):       # Clipping gradients for stability
        """Gradient clipping to the range [10, 10]."""
        parameters = list(filter(lambda p: p.grad is not None, self.machine.parameters()))
        for p in parameters:
            p.grad.data.clamp_(-10, 10)

    def train_model(self):
        # Here, the model is optimized using Cross Entropy Loss, however, it is evaluated using Number of error bits in predction and actual labels (cost)
        loss_list = []
        seq_length = []
        save_batch = 500
        last_batch = 0

        for j in range(self.num_epoch):
            for batch_num, X, Y, mask in self.get_training_data():
                self.optimizer.zero_grad()                      # Making old gradients zero before calculating the fresh ones
                self.machine.initialization(self.batch_size)    # Initializing states
                Y_out = torch.zeros(X.shape).cuda()

                # Feeding the DNC network all the data first and then predicting output
                # by giving zero vector as input and previous read states and hidden vector
                # and thus training vector this way to give outputs matching the labels

                X, Y, mask = X.cuda(), Y.cuda(), mask.cuda()       # Sending to CUDA device

                for i in range(X.shape[0]):
                    Y_out[i, :, :], _ = self.machine(X[i])

                loss = self.calc_loss(Y_out, Y, mask)
                loss.backward()
                self.clip_grads()
                self.optimizer.step()

                loss_list += [loss.item()]
                seq_length += [Y.shape[0]]

                if (batch_num % save_batch) == 0:
                    self.save_model(j, batch_num)

                last_batch = batch_num
                print("Epoch: " + str(j) + "/" + str(self.num_epoch) + ", Batch: " + str(batch_num) + "/" + str(self.num_batches) + ", Loss: " + str(loss.item()))
            self.save_model(j, last_batch)
    
    def save_model(self, curr_epoch, curr_batch):
        # Here 'start_epoch' and 'start_batch' params below are the 'epoch' and 'batch' number from which to start training after next model loading
        # Note: It is recommended to start from the 'start_epoch' and not 'start_epoch' + 'start_batch', because batches are formed randomly
        
        if not os.path.exists("Saved_Models/" + self.name):
            os.mkdir("Saved_Models/" + self.name)
        state_dic = {'task_name': self.name, 'start_epoch': curr_epoch + 1, 'start_batch': curr_batch + 1, 'state_dict': self.machine.state_dict(), 'optimizer_dic' : self.optimizer.state_dict()}
        filename = "Saved_Models/" + self.name + "/" + self.name + "_" + str(curr_epoch) + "_" + str(curr_batch) + "_saved_model.pth.tar"
        torch.save(state_dic, filename)

    def load_model(self, option, epoch, batch):
        path = "Saved_Models/" + self.name + "/" + self.name + "_" + str(epoch) + "_" + str(batch) + "_saved_model.pth.tar"
        if option == 1:             # Loading for training
            checkpoint = torch.load(path)
            self.machine.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_dic'])
        else:                       # Loading for testing
            checkpoint = torch.load(path)
            self.machine.load_state_dict(checkpoint['state_dict'])
            self.machine.eval()
