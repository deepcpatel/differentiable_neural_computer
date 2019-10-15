# DNC memory module
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

class memory_unit(nn.Module):

    def __init__(self, N, M, memory_init=None):

        super(memory_unit, self).__init__()

        self.N = N      # Number of Memory cells
        self.M = M      # Single Memory cell length

        # Registering Memory Initialization matrix into state_dict
        self.register_buffer('memory_init', torch.Tensor(N, M))

        # Memory Initialization
        stdev = 1.0 / (np.sqrt(N + M))

        # Following snippet allows user to exclusively give initialization values to the memory
        # Otherwise it initializes automatically
        if memory_init == None:
            nn.init.uniform_(self.memory_init, -stdev, stdev)   # Memory size is (N, M)
        else:
            self.memory_init = memory_init.clone()

    def memory_size(self):
        return self.N, self.M

    def reset_memory(self, batch_size):
        self.batch_size = batch_size
        self.memory = self.memory_init.clone().repeat(batch_size, 1, 1)

    def memory_read(self, W):                                               # Assuming shape of Memory -> (batch_size, N, M)
        """
        Input   : 

        W       : Read Weights          -> (batch_size x N)
        """
        return torch.bmm(W.unsqueeze(1), self.memory).squeeze(1)            # Out : (batch_size x M) size vector

    def memory_write(self, W, e, a):
        """
        Input   : 

        W       : Write Weights         -> (batch_size x N)
        e       : Erase Vector          -> (batch_size x M)
        a       : Write Vector          -> (batch_size x M)
        """
        erase_mat = torch.bmm(W.unsqueeze(-1), e.unsqueeze(1))      # Out : (batch_size x N x M) matrix
        add_mat = torch.bmm(W.unsqueeze(-1), a.unsqueeze(1))        # Out : (batch_size x N x M) matrix
        self.memory = self.memory * (1 - erase_mat) + add_mat       # Out : (batch x N x M) matrix

    def access_memory_write(self, k, beta, g_w, g_a, alloc_weights):      # Returns the weight vector to access memory for write handle
        """
        Input           : 

        k               : Key vector for matching       -> (batch_size x M)
        beta            : Constant for strength focus   -> (batch_size x 1)
        g_w             : Interpolation Write gate      -> (batch_size x 1)
        g_a             : Allocation Gate               -> (batch_size x 1)
        alloc_weights   : Allocation Weights            -> (batch_size x N)
        """
        # Content based addressing
        W_c = self._content_focusing(k, beta)           # Out : (batch_size x N) vector

        # Location based addressing
        W = self._gating(g_w, g_a, alloc_weights, W_c)  # Out : (batch_size x N) vector
        return W                                        # Out : (batch_size x N) vector

    def _read_mode_interpolation(self, f, b, W_c, r_mode):  # Performs Interpolation of Forward, Backward and Content vectors to generate Read weights
        """
        Input           : 

        f               : Forward Temporal Weights      -> 'num_write_heads' sized list of (batch_size x N) tensors
        b               : Backward Temporal Weights     -> 'num_write_heads' sized list of (batch_size x N) tensors
        W_c             : Content Similarity Weights    -> (batch_size x N)
        r_mode          : Reading Mode Vector           -> (batch_size x num_read_mode)
        """
        i = 0
        W = torch.zeros(W_c.shape)

        for forward in f:
            W = W + r_mode[:,i].unsqueeze(-1)*forward
            i += 1

        W = W + r_mode[:,i].unsqueeze(-1)*W_c
        i += 1

        for backward in b:
            W = W + r_mode[:,i].unsqueeze(-1)*backward
            i += 1

        return W    # Out : (batch_size x N) vector

    def access_memory_read(self, k, beta, f, b, r_mode):      # Returns the weight vector to access memory for read handle
        """
        Input           : 

        k               : Key vector for matching       -> (batch_size x M)
        beta            : Constant for strength focus   -> (batch_size x 1)
        f               : Forward Temporal Weights      -> 'num_write_heads' sized list of (batch_size x N) tensors
        b               : Backward Temporal Weights     -> 'num_write_heads' sized list of (batch_size x N) tensors
        r_mode          : Reading Mode Vector           -> (batch_size x num_read_mode)
        """
        # Content based addressing
        W_c = self._content_focusing(k, beta)                   # Out : (batch_size x N) vector

        # Read Mode Interpolation
        W = self._read_mode_interpolation(f, b, W_c, r_mode)    # Out : (batch_size x N) vector
        return W                                                # Out : (batch_size x N) vector

    def _content_focusing(self, key, beta):
        """
        Input   : 

        k       : Key vector for matching       -> (batch_size x M)
        beta    : Constant for strength focus   -> (batch_size x 1)
        """
        similarity_vector = F.cosine_similarity(key.unsqueeze(1) + 1e-16, self.memory + 1e-16, dim = 2) # We are adding some offset to inputs to avoid zero output 
        temp_vec = beta*similarity_vector      # similarity_vector -> dim : (batch_size x N)
        return F.softmax(temp_vec, dim = 1)

    def _gating(self, g_w, g_a, alloc_weights, W_c):    # Performs gating/interpolation
        """
        Input           : 

        g_w             : Interpolation Write gate      -> (batch_size x 1)
        g_a             : Allocation Gate               -> (batch_size x 1)
        alloc_weights   : Allocation Weights            -> (batch_size x N)
        W_c             : Content Similarity Weights    -> (batch_size x N)
        """
        return g_w*(g_a*alloc_weights + (1-g_a)*W_c)