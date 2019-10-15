# Training DNC

import sys
import time
import torch
import random
import numpy as np

# torch.autograd.set_detect_anomaly(True) # Setting Anomaly Detection True for finding bad operations

####### Following function is adapted from the NTM implemention by loudinthecloud on Github ##########
def random_seed():
    seed = int(time.time()*10000000)
    random.seed(seed)
    np.random.seed(int(seed/10000000))      # NumPy seed Range is 2**32 - 1 max
    torch.manual_seed(seed)
##########################################################################################################

def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "1":
            if sys.argv[2] == "GPU" and torch.cuda.is_available():  # Checking if GPU Request is given or not and availability of CUDA
                from tasks.copy_task_GPU import task_copy
            elif sys.argv[2] == "CPU":
                from tasks.copy_task import task_copy
            else:
                print("Please specify the run device (GPU/CPU)")
                exit()
            c_task = task_copy()                    # Initialization of the Copy Task
            print("\nStarting Copy Task for DNC\n")
        elif sys.argv[1] == "2":
            if sys.argv[2] == "GPU" and torch.cuda.is_available():  # Checking if GPU Request is given or not and availability of CUDA
                from tasks.babi_task_GPU import task_babi
            elif sys.argv[2] == "CPU":
                from tasks.babi_task import task_babi
            else:
                print("Please specify the run device (GPU/CPU)")
                exit()
            c_task = task_babi()                    # Initialization of the bAbI Task
            print("\nStarting bAbI Question Answering Task for DNC\n")
        else:
            print("Task does not exist...")
            exit()
    else:
        print("Error: Enter the task for DNC training!\n1:Copy \n2:bAbI\n")
        exit()

    # Random Seed
    random_seed()

    c_task.init_dnc()
    c_task.init_loss()
    c_task.init_optimizer()

    c_task.train_model()

    '''
    if c_task.get_task_name() == "copy_task" or c_task.get_task_name() == "copy_task_GPU":
        loss, cost, inp, labels, prediction = c_task.test_model()
    elif c_task.get_task_name() == "bAbI_task" or c_task.get_task_name() == "bAbI_task_GPU":
        accuracy = c_task.test_model()
    '''

    print("Training Completed!")

if __name__ == '__main__':
    main()