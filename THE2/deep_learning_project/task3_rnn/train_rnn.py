"""
Training script for RNN character-level prediction
"""

import torch
import torch.nn.functional as F
from .rnn_model import ManualRNN, SimpleRNN


def prepare_data(text="Deep Learning"):
    """
    Prepare character-level data for RNN training
    
    Args:
        text (str): Input text for character-level modeling
        
    Returns:
        dict: Dictionary containing processed data and metadata
    """
    # Create vocabulary
    chars = sorted(list(set(text)))
    char2idx = {ch: i for i, ch in enumerate(chars)}
    idx2char = {i: ch for i, ch in enumerate(chars)}
    
    # Create input and target sequences
    input_seq = text[:-1]  # All characters except the last
    target_seq = text[1:]  # All characters except the first
    
    # Model parameters
    V = len(chars)  # Vocabulary size
    input_size = V  # One-hot input vector size
    H = 16  # Hidden units
    output_size = V  # Output size
    seq_len = len(input_seq)
    
    print(f"Text: '{text}'")
    print(f"Unique characters: {chars}")
    print(f"Vocabulary size: {V}")
    print(f"Character to index mapping: {char2idx}")
    print(f"Input sequence: '{input_seq}'")
    print(f"Target sequence: '{target_seq}'")
    print(f"Sequence length: {seq_len}")
    
    # Convert to one-hot and indices
    def one_hot(idx, size):
        vec = torch.zeros(size)
        vec[idx] = 1.0
        return vec
    
    inputs = [one_hot(char2idx[ch], V) for ch in input_seq]
    targets = torch.tensor([char2idx[ch] for ch in target_seq], dtype=torch.long)
    
    return {
        'inputs': inputs,
        'targets': targets,
        'char2idx': char2idx,
        'idx2char': idx2char,
        'vocab_size': V,
        'hidden_size': H,
        'seq_len': seq_len,
        'input_seq': input_seq,
        'target_seq': target_seq
    }


def train_manual_rnn():
    """
    Train the manual RNN implementation and verify gradients
    """
    # Prepare data
    data = prepare_data("Deep Learning")
    
    # Initialize manual RNN
    manual_rnn = ManualRNN(
        input_size=data['vocab_size'],
        hidden_size=data['hidden_size'],
        output_size=data['vocab_size']
    )
    
    # Forward pass
    logits_list, final_hidden, hidden_states = manual_rnn.forward(data['inputs'])
    
    # Compute loss
    loss_manual = manual_rnn.compute_loss(logits_list, data['targets'])
    print(f"\nManual RNN Loss: {loss_manual.item():.4f}")
    
    # Compute gradients explicitly
    print("\nComputing gradients explicitly using torch.autograd...")
    manual_grads = manual_rnn.compute_gradients(loss_manual)
    
    # Print gradient information
    param_names = ['W_xh', 'W_hh', 'b_xh', 'b_hh', 'W_hy', 'b_y']
    print("\nGradient shapes:")
    for name, grad in zip(param_names, manual_grads):
        print(f"{name}: {grad.shape}")
    
    # Verify against PyTorch RNN
    print("\n" + "="*50)
    print("Verifying against PyTorch RNN implementation")
    print("="*50)
    
    # Create PyTorch RNN with same weights
    pytorch_rnn = SimpleRNN(
        input_size=data['vocab_size'],
        hidden_size=data['hidden_size'],
        output_size=data['vocab_size']
    )
    pytorch_rnn.set_weights_from_manual(manual_rnn)
    
    # Forward pass with PyTorch RNN
    logits_pytorch = pytorch_rnn(data['inputs'])
    log_probs_pytorch = F.log_softmax(logits_pytorch, dim=1)
    loss_pytorch = F.nll_loss(log_probs_pytorch, data['targets'])
    
    # Backward pass
    loss_pytorch.backward()
    
    # Compare gradients
    pytorch_grads = []
    for name, param in pytorch_rnn.named_parameters():
        if param.grad is not None:
            pytorch_grads.append(param.grad)
    
    print("\nGradient comparison:")
    grade = 0
    for name, manual_grad, pytorch_grad in zip(param_names, manual_grads, pytorch_grads):
        diff = (manual_grad - pytorch_grad).abs().max()
        print(f"Δ {name}: max abs diff = {diff:.6e}")
        if diff <= 1e-4:
            grade += 5
        else:
            print("  ❗ Significant difference detected")
    
    print(f"\nGrade: {grade}/30")
    
    # Generate sample text
    print("\n" + "="*50)
    print("Sample text generation")
    print("="*50)
    
    # Start with the first character
    current_char = data['input_seq'][0]
    generated = current_char
    h = torch.zeros(data['hidden_size'])
    
    # Generate 20 characters
    for _ in range(20):
        # Convert to one-hot
        x = torch.zeros(data['vocab_size'])
        x[data['char2idx'][current_char]] = 1.0
        
        # Forward pass for single character
        h = torch.tanh(manual_rnn.W_xh @ x + manual_rnn.b_xh + 
                      manual_rnn.W_hh @ h + manual_rnn.b_hh)
        logits = manual_rnn.W_hy @ h + manual_rnn.b_y
        
        # Sample from distribution
        probs = F.softmax(logits, dim=0)
        idx = torch.multinomial(probs, 1).item()
        current_char = data['idx2char'][idx]
        generated += current_char
    
    print(f"Generated text: '{generated}'")
    
    return manual_rnn, manual_grads, grade


if __name__ == "__main__":
    manual_rnn, gradients, grade = train_manual_rnn()