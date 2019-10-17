## Differentiable Neural Computer (DNC) implementation in PyTorch

• My implementation of Differentiable Neural Computer (DNC) in PyTorch.<br/>
• DNC is introduced in the paper [Hybrid computing using a neural network with dynamic external memory](https://www.nature.com/articles/nature20101).<br/>
• Currently, **bAbI Question Answering task** and **Pattern Copy task** is implemented for CPU and GPU each.<br/>
• Although I have tested the code thoroughly, bugs may persist. In that case you are encouraged to report them.<br/>

#### Platform:
The code is written in `Python 3.6` in `Ubuntu 18.04` Operating System.

#### Libraries required:
NumPy <br/>
PyTorch

#### Train the model by writing following in the terminal:

`python3 train.py opt1 opt2`

```
opt1: 1). 1 for Copy_Task
      2). 2 for bAbI_Task

opt2: 1). GPU to run code on GPU
      2). CPU to run code normally
```

#### Test the model by writing following in the terminal:

`python3 test.py opt1 opt2 opt3 opt4`

```
opt1: 1). 1 for Copy_Task
      2). 2 for bAbI_Task

opt2: 1). GPU to run code on GPU
      2). CPU to run code normally

opt3: Last Epoch number till the model was trained (Not Applicable for Copy Task. Any value is fine)

Opt4: Last Batch Number till the model was trained 
```

#### References:
1). https://github.com/loudinthecloud/pytorch-ntm <br/>
2). https://github.com/bgavran/DNC
