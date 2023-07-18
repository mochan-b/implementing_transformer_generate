import torch
from transformer_model import TransformerModel
from wikitext2_tokenizer import WikiText2Tokenizer

# Load the tokenizer and create the tokens
tokenizer = WikiText2Tokenizer()
tokens = tokenizer.encode('Once upon a time')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ntokens = len(tokenizer.vocab)  # the size of vocabulary
emsize = 200  # embedding dimension
d_hid = 200  # dimension of the feedforward network model in ``nn.TransformerEncoder``
nlayers = 2  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
nhead = 2  # number of heads in ``nn.MultiHeadAttention``
dropout = 0.2  # dropout probability
model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)
model.load_state_dict(torch.load("transformer_model.pt"))

# Number of texts to generate
max_length = 25

# Change the tokens to input tensors. Batch size is 1 and is the second dimension
input_length = len(tokens)
input = torch.tensor(tokens).unsqueeze(0).view(input_length, 1).to(device)

# Generate text until the output length (which includes the context length)
for i in range(max_length):
    # Create a mask for the input
    input_length = input.shape[0]
    src_mask = torch.ones(input_length, input_length).to(device)
    output = model(input, src_mask)

    # Get the next token probabilities as the last column of the output tensor
    next_token_probs = output[-1, 0, :]
    next_token_softmax = torch.nn.functional.softmax(next_token_probs, dim=0)
    next_token = torch.multinomial(next_token_softmax, num_samples=1)

    # Append the next token to the input
    input = torch.cat([input, next_token.unsqueeze(0)], dim=-0)

# Decode the generated text
generated_words = tokenizer.decode(input[:, 0].tolist())
generated_text = ' '.join(generated_words)
print(generated_text)
