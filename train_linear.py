from torch.utils.data import Dataset, DataLoader
import torch
from resemblyzer import VoiceEncoder, preprocess_wav
import numpy
from pathlib import Path
def vectorExtract(audio1_path,audio2_path):
    wav1 = preprocess_wav(Path(audio1_path))
    wav2 = preprocess_wav(Path(audio2_path))

    encoder = VoiceEncoder()

    embed1 = encoder.embed_utterance(wav1)
    embed2 = encoder.embed_utterance(wav2)

    return numpy.concatenate([embed1,embed2])

class AudioDataset(Dataset):

    def __init__(self, lines):
        self.lines = lines

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):

        audio1_path,audio2_path,label = self.lines[idx].split(",")
        audio1_path= "/home/src/DATA/dataset/" + audio1_path
        audio2_path= "/home/src/DATA/dataset/" + audio2_path

        vec = vectorExtract(audio1_path,audio2_path)

        return vec, torch.Tensor(int(label))

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


train_lines = []
with open("/home/src/DATA/pair.txt", "r", encoding="utf-8") as f:
  train_lines = f.read().splitlines()

train_loader = DataLoader(AudioDataset(train_lines), batch_size=100, shuffle=True, pin_memory=True)



D_in, H, D_out = 512, 100, 1
epoch = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TwoLayerNet(D_in, H, D_out).to(device)
model.parameters()
loss_fn = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
for t in range(epoch):
    print("epoch: "+str(epoch))
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), "train_linear_out.pt")
