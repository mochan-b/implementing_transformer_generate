import os.path
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import WikiText2


# Tokenizer that builds the vocabulary from wikitext2 dataset and basic_english tokenizer
class WikiText2Tokenizer:
    def __init__(self):
        self.tokenizer = get_tokenizer('basic_english')
        self.load_vocab('wikitext2_vocab.pt')

    # Load a vocab from disk or build a new one from the dataset
    def load_vocab(self, vocab_path):
        if os.path.exists(vocab_path):
            self.vocab = torch.load(vocab_path)
        else:
            self.vocab = build_vocab_from_iterator(map(self.tokenizer, WikiText2(split='train')), specials=['<unk>'])
            self.vocab.set_default_index(self.vocab['<unk>'])
            torch.save(self.vocab, vocab_path)

    def encode(self, input):
        return self.vocab(self.tokenizer(input))

    def decode(self, input):
        return self.vocab.lookup_tokens(input)


if __name__ == '__main__':
    tokenizer = WikiText2Tokenizer()
    tokens = tokenizer.encode('Hi world!')
    print(tokens)
    original_text = tokenizer.decode(tokens)
    print(original_text)
