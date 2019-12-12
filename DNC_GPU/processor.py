# Processor for Differentiable Neural Computer
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from .head import DNC_read_head, DNC_write_head
from .controller import controller

class processor(nn.Module):
    def __init__(self, num_inputs, output_size, memory_M, memory_N, num_read_heads, num_write_heads, controller_size, controller_layers):
        # Parameters:
        # num_inputs -> Size of input data
        # output_size -> Controller Output Size
        # memory_M -> Width of each strip of Memory
        # memory_M -> Number of memory cells
        # num_read_heads -> Number of Read heads to be created
        # num_write_heads -> Number of Write heads to be created
        # controller_size -> Size of LSTM Controller output/state
        # controller_layers -> Number of layers in LSTM Network

        super(processor, self).__init__()

        # Initializing read head values
        self.num_read_heads = num_read_heads
        self.num_write_heads = num_write_heads
        self.M = memory_M
        self.N = memory_N

        # Creating the Controller
        self.controller = controller(num_inputs + self.M*self.num_read_heads, controller_size, controller_layers)
        self.controller.cuda()  # Sending controller to CUDA cores

        # Creating Read and Write heads
        self.heads = nn.ModuleList([])

        for i in range(self.num_read_heads):
            self.heads += [DNC_read_head(self.N, self.M)]

        for i in range(self.num_write_heads):
            self.heads += [DNC_write_head(self.N, self.M)]

        # Initializing Misc. variables
        self.init_r = []
        self.usage = []                 # Stores the Usage Vectors for all the write heads
        self.tempo_links = []           # Stores the Temporal Linkages matrices L for all the write heads
        self.prec_weights = []          # Stores the Precedence weight vectors P for all the write heads
        self.num_read_mode = 1 + 2*self.num_write_heads        # 1 for content lookup and 2*num_write_head for forward and backward operation for each write head

        # 1). Buffer are Tensors with require_grads() False
        # 2). But, in PyTorch, Tensors created; by default have require_grads() False
        # 3). Therefore, I think that both are same
        # 4). Variables in PyTorch are replaced with Tensors from verson 0.4.0
        # 5). Buffers are useful for storing values of an entity into state_dict (done using register_buffer() method) but does not require calculation of gradients on the entity

        for head in self.heads:
            if head.head_type() == 'R':
                temp = torch.randn(1, self.M)*0.01
                # self.register_buffer("read_bias_" + str(self.num_read_heads), temp.data)
                self.init_r += [temp]
            else:
                self.usage += [torch.zeros(1, self.N)]                  # Initializing the usage vector to zero
                self.tempo_links += [torch.zeros(1, self.N, self.N)]    # Initializing the temporal linkages matrices
                self.prec_weights += [torch.zeros(1, self.N)]           # Initializing the precedence weights

        assert self.num_read_heads > 0, "Read Heads must be atleast 1"

        # Creating Linear Layer Transformation for getting Output
        self.proc = nn.Linear(controller_size + self.num_read_heads*self.M, output_size)

        # Creating Linear Layer Transformation for getting Parameters
        self.write_vectors = nn.Linear(controller_size, self.num_write_heads*self.M)
        self.erase_vectors = nn.Linear(controller_size, self.num_write_heads*self.M)
        self.free_gate = nn.Linear(controller_size, self.num_read_heads)
        self.allocation_gate = nn.Linear(controller_size, self.num_write_heads)
        self.write_gate = nn.Linear(controller_size, self.num_write_heads).cuda()
        self.read_mode = nn.Linear(controller_size, self.num_read_heads*self.num_read_mode)
        self.write_keys = nn.Linear(controller_size, self.num_write_heads*self.M)
        self.write_strengths = nn.Linear(controller_size, self.num_write_heads)
        self.read_keys = nn.Linear(controller_size, self.num_read_heads*self.M)
        self.read_strengths = nn.Linear(controller_size, self.num_read_heads)

        # Initializing the Parameters for Linear Layers
        nn.init.xavier_uniform_(self.proc.weight, gain=1)
        nn.init.normal_(self.proc.bias, std=0.01)

        nn.init.xavier_uniform_(self.write_vectors.weight, gain=1)
        nn.init.normal_(self.write_vectors.bias, std=0.01)

        nn.init.xavier_uniform_(self.erase_vectors.weight, gain=1)
        nn.init.normal_(self.erase_vectors.bias, std=0.01)

        nn.init.xavier_uniform_(self.free_gate.weight, gain=1)
        nn.init.normal_(self.free_gate.bias, std=0.01)

        nn.init.xavier_uniform_(self.allocation_gate.weight, gain=1)
        nn.init.normal_(self.allocation_gate.bias, std=0.01)

        nn.init.xavier_uniform_(self.write_gate.weight, gain=1)
        nn.init.normal_(self.write_gate.bias, std=0.01)

        nn.init.xavier_uniform_(self.read_mode.weight, gain=1)
        nn.init.normal_(self.read_mode.bias, std=0.01)

        nn.init.xavier_uniform_(self.write_keys.weight, gain=1)
        nn.init.normal_(self.write_keys.bias, std=0.01)

        nn.init.xavier_uniform_(self.write_strengths.weight, gain=1)
        nn.init.normal_(self.write_strengths.bias, std=0.01)

        nn.init.xavier_uniform_(self.read_keys.weight, gain=1)
        nn.init.normal_(self.read_keys.bias, std=0.01)

        nn.init.xavier_uniform_(self.read_strengths.weight, gain=1)
        nn.init.normal_(self.read_strengths.bias, std=0.01)

    def create_new_state(self, batch_size):     # Re-creates the New States
        init_r = [r.clone().repeat(batch_size, 1).cuda() for r in self.init_r]
        controller_state = self.controller.create_hidden_state(batch_size)
        heads_state = [head.create_new_state(batch_size) for head in self.heads]
        usage = [u.repeat(batch_size, 1).cuda() for u in self.usage]                  # Extending the Usage Vector size to accomodate multiple batches
        prec_weights = [p.repeat(batch_size, 1).cuda() for p in self.prec_weights]    # Extending the Precedence weight vector size to accomodate multiple batches
        tempo_links = [l.repeat(batch_size, 1, 1).cuda() for l in self.tempo_links]   # Extending the Temporal link vector size to accomodate multiple batches

        return init_r, controller_state, heads_state, usage, prec_weights, tempo_links      

    def param_operations(self, inp):               # Calculating the head parameters based on the input from the LSTM controller
        output = {}

        output['write_vectors'] = (self.write_vectors(inp)).view(-1, self.num_write_heads, self.M)                        # Dim: (batch_size x num_write_heads x M)
        output['erase_vectors'] = torch.sigmoid(self.erase_vectors(inp)).view(-1, self.num_write_heads, self.M)           # Dim: (batch_size x num_write_heads x M)
        output['free_gate'] = torch.sigmoid(self.free_gate(inp))                                                          # Dim: (batch_size x num_read_heads)
        output['allocation_gate'] = torch.sigmoid(self.allocation_gate(inp))                                              # Dim: (batch_size x num_write_heads)
        output['write_gate'] = torch.sigmoid(self.write_gate(inp))                                                        # Dim: (batch_size x num_write_heads)
        output['read_mode'] = F.softmax((self.read_mode(inp)).view(-1, self.num_read_heads, self.num_read_mode), dim=2)   # Dim: (batch_size x num_read_heads x num_read_mode)
        output['write_keys'] = (self.write_keys(inp)).view(-1, self.num_write_heads, self.M)                              # Dim: (batch_size x num_write_heads x M)
        output['write_strengths'] = 1 + F.softplus(self.write_strengths(inp))                                             # Dim: (batch_size x num_write_heads)
        output['read_keys'] = (self.read_keys(inp)).view(-1, self.num_read_heads, self.M)                                 # Dim: (batch_size x num_read_heads x M)
        output['read_strengths'] = 1 + F.softplus(self.read_strengths(inp))                                               # Dim: (batch_size x num_read_heads)
        return output

    def calc_alloc_weights(self, head_params, prev_head_weights, prev_usage):   # Calculates allocation weights and returns them
        # Initializing allocation weights
        alloc_weights = []
        new_usage = []
        for u in prev_usage:
            alloc_weights.append(torch.zeros(u.shape).cuda())       # alloc_weights is a list of size 'num_write_heads' 
                        
        # Calculating psi -> Memory Retention Vector
        i = 0
        temp = head_params['free_gate'].unsqueeze(1)                # temp Dim : (batch_size x 1 x no_read_heads)
        psi = torch.ones(temp.shape[0], self.N).cuda()
        
        for head, prev_weights in zip(self.heads, prev_head_weights):
            if head.head_type() == 'R':
                psi = psi*(1 - temp[:,:,i]*prev_weights)            # psi Dim : (batch_size x N)
                i += 1 

        # Calculating Usage vector
        i = 0
        for head, prev_weights in zip(self.heads, prev_head_weights):
            if head.head_type() == 'W':
                temp = prev_usage[i]
                new_usage.append((temp + prev_weights - temp*prev_weights)*psi)
                i += 1

        '''
        # Calculating Usage vector
        i = 0
        for head, prev_weights in zip(self.heads, prev_head_weights):
            if head.head_type() == 'W':
                temp = self.usage[i]
                self.usage[i] = (temp + prev_weights - temp*prev_weights)*psi
                i += 1

        Note: In the above code snippet, I was getting Error "RuntimeError: Trying to backward through the graph a 
        second time, but the buffers have already been freed. Specify retain_graph=True when calling backward the 
        first time." - possibly because I have declared "self.usage" variable during the initialization, and now in 
        the first iteration of backpropogation, the program will run normally and will delete the newly constructed 
        graph after that. However, in the next iteration, it will construct the graph again, and will not find the 
        initialization in the graph, because the previous graph having the initialization operation was deleted 
        (as explained here: https://stackoverflow.com/questions/55268726/pytorch-why-does-preallocating-memory-cause-trying-to-backward-through-the-gr).
        One solution is to enable "specify retain_graph=True" in the backward() function. However, this stores all the 
        graph of the previous iterations and therefore, each succeding iteration will take more time to execute as well 
        as there will be significant overhead. Therefore, instead of initializing, I am going to pass the previous values 
        in the form of arguments of the function and will return the new values from the function, to maintain the graph.
        '''

        # Calculating Allocation Vector
        i = 0
        for u in new_usage:
            np_ar = u.cpu().data.numpy()
            args = np_ar.argsort(axis=1)
            u_sorted = u[np.arange(u.shape[0]).reshape(-1,1), args]
            diff_u_sorted = 1 - u_sorted

            # Calculating Cumulitive Multiplaction
            u_cum_prod = torch.cumprod(u_sorted, dim=1)

            alloc_weights[i][np.arange(u.shape[0]).reshape(-1,1), args] = diff_u_sorted*u_cum_prod
            i += 1
        return alloc_weights, new_usage

    def calc_temporal_linkages(self, write_weights, prev_prec_weights, prev_tempo_links):    # Calculates Temporal Linkages between write positions as described in the paper
        """
        Input               :

        write_weights       : Write Weights                 -> 'num_write_heads' sized list having each element of size (batch_size x N)
        prev_prec_weights   : Previous Precedence Weights   -> 'num_write_heads' sized list having each element of size (batch_size x N)
        prev_tempo_links    : Previous Precedence Weights   -> 'num_write_heads' sized list having each element of size (batch_size x N x N)
        """
        w_head_count = 0                    # Counter
        # K = int(self.N/2)                 # Select first K sorted elements. This feature is currently not implemented. Idea: Instead of fixed K, it maybe be learned by the network.
        new_prec_weights = []               # List to store new precedence weights
        new_tempo_links = []                # List to store new Temporal Links
        idx = torch.arange(self.N).cuda()   # Range of Indices

        for head in self.heads:
            if head.head_type() == 'W':

                # Calculating Sparse Precedence Weights for Sparse linkage matrix
                p = prev_prec_weights[w_head_count].unsqueeze(1)                                    # dim : (batch_size x 1 x N)      
                # Calculating Sparse Write Weights for Sparse linkage matrix
                w = write_weights[w_head_count].unsqueeze(-1)                                       # dim : (batch_size x N x 1)
                # Calculating Linkage Matrix
                l = prev_tempo_links[w_head_count]                                                  # dim : (batch_size x N x N)

                # Transposing w
                w_t = torch.transpose(w, 1, 2)                                                      # dim : (batch_size x 1 x N)

                # Calculating New Temporal Linkage Matrix L
                temp_l = (1 - (w + w_t))*l + torch.bmm(w, p)
                temp_l[:, idx, idx] = 0                                                             # Making diagonal elements zero
                new_tempo_links.append(temp_l)

                # Calculating Precedence weights for next step (Calculating P_t)
                new_prec_weights.append((1 - torch.sum(write_weights[w_head_count], dim = 1).reshape(-1, 1))*prev_prec_weights[w_head_count] + write_weights[w_head_count])    # dim: (batch_size x N)
                w_head_count += 1
        return new_prec_weights, new_tempo_links
        
    def forward(self, X, prev_state, memory):   # X dimensions -> (batch_size x num_inputs)
        
        # Previous State Unpacking:
        prev_read, prev_controller_state, prev_head_weights, prev_usage, prev_prec_weights, prev_tempo_links = prev_state

        # prev_read[i] -> batch_size x M
        # prev_read -> batch_size x (M*no_read_heads) 
        # prev_head_weights -> (num_read_heads + num_write_heads) sized list of (batch_size x N) sized tensors

        # Making input for controller
        inp = torch.cat([X] + prev_read, dim = 1)   # inp -> (batch_size x (num_inputs + (no_read_heads * M)))
        c_output, c_state = self.controller(inp, prev_controller_state) # Getting embeddings from controller

        # Calculating head parameters based on the controller output
        head_params = self.param_operations(c_output)

        # Calculating Allocation Weights
        head_params['alloc_weights'], new_usage = self.calc_alloc_weights(head_params, prev_head_weights, prev_usage)  # 'num_write_heads' sized list having each element of size (batch_size x N)

        # Writing into the Memory
        write_weights = []  # To store write weights
        w_head_count = 0    # Counter

        for head in self.heads:
            if head.head_type() == 'W':
                weights = head(head_params, w_head_count, memory)
                write_weights += [weights]
                w_head_count += 1

        # Writing Temporal Linkage Matrix L
        new_prec_weights, new_tempo_links = self.calc_temporal_linkages(write_weights, prev_prec_weights, prev_tempo_links)

        # Reading from the Memory
        read_vec = []       # To store read vectors
        read_weights = []   # To store read weights
        r_head_count = 0    # Counter

        for head, prev_weights in zip(self.heads, prev_head_weights):
            if head.head_type() == 'R':
                weights, r_vec = head(head_params, prev_weights, new_tempo_links, r_head_count, memory)
                read_weights += [weights]
                read_vec += [r_vec]
                r_head_count += 1

        # Re-arranging the weights into larger list
        head_weights = []   # To store all the weights
        r_head_count = 0    # Read head counter
        w_head_count = 0    # Write head counter

        for head in self.heads:
            if head.head_type() == 'R':
                head_weights.append(read_weights[r_head_count])
                r_head_count += 1
            else:
                head_weights.append(write_weights[w_head_count])
                w_head_count += 1

        # Packing State Vectors for Next step
        curr_state = (read_vec, c_state, head_weights, new_usage, new_prec_weights, new_tempo_links)

        # Generating Output
        inp2 = torch.cat([c_output] + read_vec, dim = 1)
        out = self.proc(inp2)                               # Passing the read vectors and controller output to the linear layer to generate final output
        return out, curr_state