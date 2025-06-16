"""
RNN implementation using torch.autograd for character-level prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ManualRNN:
    """
    Manual RNN implementation using torch.autograd for gradient computation
    
    This class implements a character-level RNN with explicit gradient calculation
    for educational purposes.
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize RNN parameters
        
        Args:
            input_size (int): Size of input vocabulary (number of unique characters)
            hidden_size (int): Number of hidden units
            output_size (int): Size of output vocabulary (same as input_size for char prediction)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights with small random values
        self.W_xh = torch.randn(hidden_size, input_size, requires_grad=True) * 0.1
        self.b_xh = torch.zeros(hidden_size, requires_grad=True)
        
        self.W_hh = torch.randn(hidden_size, hidden_size, requires_grad=True) * 0.1
        self.b_hh = torch.zeros(hidden_size, requires_grad=True)
        
        self.W_hy = torch.randn(output_size, hidden_size, requires_grad=True) * 0.1
        self.b_y = torch.zeros(output_size, requires_grad=True)
        
        # Collect all parameters
        self.parameters = [self.W_xh, self.b_xh, self.W_hh, self.b_hh, self.W_hy, self.b_y]
    
    def forward(self, inputs, initial_hidden=None):
        """
        Forward pass through the RNN
        
        Args:
            inputs (list): List of one-hot encoded input tensors
            initial_hidden (torch.Tensor): Initial hidden state (default: zeros)
            
        Returns:
            tuple: (logits_list, final_hidden_state)
        """
        if initial_hidden is None:
            h = torch.zeros(self.hidden_size)
        else:
            h = initial_hidden
            
        logits_list = []
        hidden_states = []
        
        # Process each time step
        for x_t in inputs:
            # Update hidden state: h_t = tanh(W_xh @ x_t + b_xh + W_hh @ h_{t-1} + b_hh)
            h = torch.tanh(self.W_xh @ x_t + self.b_xh + self.W_hh @ h + self.b_hh)
            hidden_states.append(h)
            
            # Calculate output logits: s_t = W_hy @ h_t + b_y
            s_t = self.W_hy @ h + self.b_y
            logits_list.append(s_t)
        
        return logits_list, h, hidden_states
    
    def compute_loss(self, logits_list, targets):
        """
        Compute negative log-likelihood loss
        
        Args:
            logits_list (list): List of output logits from forward pass
            targets (torch.Tensor): Target indices
            
        Returns:
            torch.Tensor: Scalar loss value
        """
        # Stack logits into tensor
        logits = torch.stack(logits_list)
        
        # Compute log probabilities and loss
        log_probs = F.log_softmax(logits, dim=1)
        loss = F.nll_loss(log_probs, targets)
        
        return loss
    
    def compute_gradients(self, loss):
        """
        Compute gradients explicitly using torch.autograd
        
        Args:
            loss (torch.Tensor): Loss value from forward pass
            
        Returns:
            list: List of gradients for each parameter
        """
        gradients = []
        
        for param in self.parameters:
            grad = torch.autograd.grad(loss, param, retain_graph=True)[0]
            gradients.append(grad)
            
        return gradients


class SimpleRNN(nn.Module):
    """
    PyTorch RNN implementation for comparison
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn_cell = nn.RNNCell(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x_seq):
        h = torch.zeros(self.hidden_size)
        logits = []
        for x in x_seq:
            h = self.rnn_cell(x, h)
            y = self.output_layer(h)
            logits.append(y)
        return torch.stack(logits)
    
    def set_weights_from_manual(self, manual_rnn):
        """
        Copy weights from manual RNN implementation for comparison
        """
        with torch.no_grad():
            self.rnn_cell.weight_ih.copy_(manual_rnn.W_xh)
            self.rnn_cell.weight_hh.copy_(manual_rnn.W_hh)
            self.rnn_cell.bias_ih.copy_(manual_rnn.b_xh)
            self.rnn_cell.bias_hh.copy_(manual_rnn.b_hh)
            self.output_layer.weight.copy_(manual_rnn.W_hy)
            self.output_layer.bias.copy_(manual_rnn.b_y)