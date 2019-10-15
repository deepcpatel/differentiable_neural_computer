# DNC Read and Write head module

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class DNC_read_head(nn.Module):     # Read Head for Memory

    def __init__(self, N, M):

        super(DNC_read_head, self).__init__()
        self.N, self.M = N, M

    def head_type(self):
        return "R"      # Head Type : (R)ead

    def create_new_state(self, batch_size): # The State holds the Weights
        return torch.zeros(batch_size, self.N).cuda()

    def forward(self, head_parameters, W_old, tempo_links, head_no, memory):
        # W_old: Previous Read Weights      ->  (batch_size x N)

        k = head_parameters['read_keys'][:,head_no,:]                           # Read Keys         ->  dim : (batch_size x M)
        beta = (head_parameters['read_strengths'][:,head_no]).unsqueeze(-1)     # Read Strength     ->  dim : (batch_size x 1)
        r_mode = head_parameters['read_mode'][:,head_no,:]                      # Read Mode         ->  dim : (batch_size x num_read_mode)

        f = []
        b = []

        for L in tempo_links:
            f += [torch.bmm(L, W_old.unsqueeze(-1)).squeeze(-1)]                    # Forward Temporal Weights    -> dim : (batch_size x N)
            b += [torch.bmm(L.transpose(1, 2), W_old.unsqueeze(-1)).squeeze(-1)]    # Backward Temporal Weights   -> dim : (batch_size x N)

        W = memory.access_memory_read(k, beta, f, b, r_mode)   # Out : (batch_size x N)
        mem_content = memory.memory_read(W)                    # Out : (batch_size x M)
        return W, mem_content

class DNC_write_head(nn.Module):    # Write Head for Memory
    
    def __init__(self, N, M):

        super(DNC_write_head, self).__init__()
        self.N, self.M = N, M

    def head_type(self):
        return "W"          # Head Type : (W)rite

    def create_new_state(self, batch_size): # The State holds the Weights
        return torch.zeros(batch_size, self.N).cuda()

    def forward(self, head_parameters, head_no, memory):
        k = head_parameters['write_keys'][:,head_no,:]                          # Write Keys        ->  dim : (batch_size x M)
        beta = (head_parameters['write_strengths'][:,head_no]).unsqueeze(-1)    # Write Strength    ->  dim : (batch_size x 1)
        g_w = (head_parameters['write_gate'][:,head_no]).unsqueeze(-1)          # Write Gate        ->  dim : (batch_size x 1)
        g_a = (head_parameters['allocation_gate'][:,head_no]).unsqueeze(-1)     # Allocation Gate   ->  dim : (batch_size x 1)
        a = head_parameters['write_vectors'][:,head_no,:]                       # Write Vector      ->  dim : (batch_size x M)
        e = head_parameters['erase_vectors'][:,head_no,:]                       # Erase Vector      ->  dim : (batch_size x M)
        alloc_weights = head_parameters['alloc_weights'][head_no]               # Allocation Weight ->  dim : (batch_size x N)

        W = memory.access_memory_write(k, beta, g_w, g_a, alloc_weights)
        memory.memory_write(W, e, a)
        return W