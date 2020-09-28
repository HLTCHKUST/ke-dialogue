import json 
class SimpleTokenizer:
    def __init__(self, vocab, load=False):
        if(load):
            self.vocab = json.load(open(vocab+"/vocab.json"))
        else:
            self.vocab = vocab
            idx = len(self.vocab.keys())
            self.vocab["<bos>"] = idx+1
            self.vocab["<|endoftext|>"] = idx+2
            self.vocab["<speaker1>"] = idx+3
            self.vocab["<speaker2>"] = idx+4
            self.vocab["<pad>"] = idx+5

        self.inverted_vocab = {int(v):k for k,v in self.vocab.items()}
        assert len(self.vocab.keys()) == len(self.inverted_vocab.keys())

    def __len__(self):
        return len(self.vocab.keys())+1

    def convert_tokens_to_ids(self,tokens):
        if(type(tokens)==list):
            return [self.vocab[tok] for tok in tokens]
        else:
            return self.vocab[tokens]
    def encode(self,text,add_special_tokens=False):
        return [self.vocab[tok] for tok in text.split()]

    def decode(self,index,skip_special_tokens=True):
        return " ".join([self.inverted_vocab[ind] for ind in index])

    def save_pretrained(self, save_dir): 
        with open(save_dir+'/vocab.json', 'w') as fp:
            json.dump(self.vocab, fp, indent=4)