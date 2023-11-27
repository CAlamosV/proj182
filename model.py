import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch import Tensor
from data_generation import *
from torchvision import transforms
from PIL import Image, ImageDraw


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)
    
    
class SimpleTransformer(nn.Module):
    def __init__(self):
        super(SimpleTransformer, self).__init__()
        self.encoder = nn.TransformerEncoderLayer(d_model=H * W, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder, num_layers=3)
        self.fc = nn.Linear(H * W, 2)  # Output is x, y position

    def forward(self, x):
        # Flatten the image and encode
        batch_size, sequence_length, _, _, _ = x.size()

        # 2D positional encoding
        # https://static.us.edusercontent.com/files/qjovJzQ2ERbedQv2ZeCZKAPS

        x = x.view(batch_size, sequence_length, -1)
        encoded = self.transformer_encoder(x)
        output = self.fc(encoded)  # Apply the linear layer to the entire sequence
        return output


def euclidean_distance_loss(output, target):
    epsilon = 1e-6  # A small constant to avoid sqrt(0)
    return torch.sqrt(torch.sum((output - target) ** 2, dim=-1) + epsilon).mean()


if __name__ == "__main__":
        

    model = SimpleTransformer()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = euclidean_distance_loss

    for epoch in range(num_epochs):
        for images, positions in dataloader:
            # Shift the positions by one to create the targets
            targets = torch.roll(positions, -1, dims=1)

            # Mask the last position in each sequence as it has no valid next position
            mask = torch.zeros_like(targets)
            mask[:, :-1, :] = 1  # Mask all but the last position

            # Forward pass
            predicted_positions = model(images)

            # Apply the mask
            masked_predicted = predicted_positions * mask
            masked_targets = targets * mask

            # Compute loss
            loss = criterion(masked_predicted, masked_targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")