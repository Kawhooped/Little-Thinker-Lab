import re, argparse, torch
from torch import nn
from pathlib import Path

def tokenize(s):
    return re.findall(r"<[A-Z_]+>|HAHAHA|BWAHAHA|hehehe|[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\w\s]", s, re.UNICODE)

class TinyMind(nn.Module):
    def __init__(self, n_vocab, d=64, h=96):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, d, padding_idx=0)
        self.gru = nn.GRU(d, h, batch_first=True)
        self.head = nn.Linear(h, n_vocab)
    def forward(self, x):
        _, h = self.gru(self.emb(x))
        return self.head(h[-1])

p = Path(__file__).parent
ckpt = torch.load(p / "littlethinker_overflow.pt", map_location="cpu", weights_only=False)
vocab = ckpt["vocab"]
stoi = {t:i for i,t in enumerate(vocab)}
model = TinyMind(len(vocab))
model.load_state_dict(ckpt["model_state"])
model.eval()

def generate(prompt, n=35, temperature=.85):
    ids = [stoi.get(t, stoi["<BOS>"]) for t in tokenize(prompt)][-48:]
    for _ in range(n):
        x = torch.tensor([[0] * (48-len(ids)) + ids])
        with torch.no_grad():
            logits = model(x)[0] / max(temperature, .05)
        nxt = int(torch.multinomial(torch.softmax(logits, dim=-1), 1))
        if vocab[nxt] == "<EOS>": break
        ids.append(nxt)
        ids = ids[-48:]
    return " ".join(vocab[i] for i in ids if vocab[i] not in ("<PAD>","<BOS>","<TURN>"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="<BOS> <LAUGH_TRIGGER> <TURN>")
    ap.add_argument("--tokens", type=int, default=35)
    a = ap.parse_args()
    print(generate(a.prompt, a.tokens))
