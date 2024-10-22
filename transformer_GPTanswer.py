import torch
import torch.nn as nn


class TransformerGenerativeNetwork(nn.Module):
    def __init__(self, config):
        super(TransformerGenerativeNetwork, self).__init__()
        self.config = config

        # Transformer decoder layers
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.config.N_n_units, nhead=8, dim_feedforward=512)
        self.transformer_decoder = nn.TransformerDecoder(
            self.transformer_decoder_layer, num_layers=6)

        # Token embedding layer (if working with tokens, such as in NLP tasks)
        self.token_embedding = nn.Embedding(self.config.vocab_size,
                                            self.config.N_n_units)

        # Output layer to project back to token space
        self.output_layer = nn.Linear(self.config.N_n_units,
                                      self.config.vocab_size)

    def forward(self, x, target_len):
        """
        x: Input token (batch_size, 1, embedding_dim) -> Typically the start token
        target_len: Length of the sequence to be generated (e.g., 8)
        """
        batch_size = x.size(0)

        # Prepare a tensor to store the generated tokens
        generated_sequence = torch.zeros(batch_size, target_len,
                                         self.config.N_n_units).to(x.device)

        # Initialize with the first token (start token)
        current_input = x  # x is (batch_size, 1, N_n_units)

        for i in range(target_len):
            # Pass the current input through the transformer decoder
            h = self.transformer_decoder(current_input, current_input)

            # Extract the output for the current token
            token_output = self.output_layer(
                h[:, -1, :])  # Use the output of the last token in the sequence

            # Convert to next token embedding (if you are generating tokens)
            next_token_embedding = self.token_embedding(
                torch.argmax(token_output, dim=-1))

            # Append the new token embedding to the input for the next iteration
            current_input = torch.cat(
                [current_input, next_token_embedding.unsqueeze(1)], dim=1)

            # Store the generated token embedding in the sequence
            generated_sequence[:, i, :] = next_token_embedding

        return generated_sequence


# Example config object
class Config:
    def __init__(self):
        self.N_n_units = 64  # Embedding dimension for the tokens
        self.vocab_size = 1000  # Size of the vocabulary (if generating tokens)


# Initialize the network
config = Config()
model = TransformerGenerativeNetwork(config)

# Example input: single token (batch of sequences, length 1)
x = torch.randint(0, config.vocab_size, (32, 1)).to(
    torch.int64)  # Batch size 32, sequence length 1 (start token)

# Convert token to embedding
x = model.token_embedding(x)

# Specify the target length of the generated sequence (e.g., 8 tokens)
target_len = 8

# Forward pass through the model to generate a sequence of length 8
output = model(x, target_len)

print("Generated Sequence Shape:",
      output.shape)  # Should output (batch_size, 8, N_n_units)
