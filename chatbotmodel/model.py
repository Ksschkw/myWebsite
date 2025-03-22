# MultipleFiles/model.py
import torch
import torch.nn as nn

class ContextAwareNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, context_size=0,dropout_prob=0.5):
        """
        Args:
            input_size: Base input features (BoW vector size)
            hidden_size: Hidden layer dimension
            num_classes: Number of intent tags
            context_size: Additional context features (default 0 for backward compatibility)
        """
        super(ContextAwareNN, self).__init__()
        total_input_size = input_size + context_size
        
        self.layer1 = nn.Linear(total_input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)  # Regularization

    def forward(self, x):
        # Dynamic context integration
        out = self.layer1(x)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.layer2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.layer3(out)  # No activation for raw logits
        return out

# Alias for backward compatibility
NeuralNet = ContextAwareNN