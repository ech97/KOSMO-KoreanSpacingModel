import json
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torchtext

parser = ArgumentParser()
parser.add_argument("--train-file", type=str, required=True)
parser.add_argument("--dev-file", type=str, required=True)
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--char-file", type=str, required=True)

class SpacingModel(nn.Module):
    def __init__(
        self, 
        vocab_size, 
        hidden_size, 
        num_classes = 3, 
        conv_activation="relu", 
        dense_activation="relu",
        kernel_and_filter_sizes = [
            (2, 8),
            (3, 8),
            (4, 8),
            (5, 8)
        ],
        dropout_rate = 0.3
        ):
        super(SpacingModel, self).__init__()

        # 5000개(vocab_size)의 단어를 각각 48차원(hidden_size)으로 Embedding 진행
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        
        layers = []
        in_channels = hidden_size
        # 2, 3, 4, 5 증가하는 Kernel size는 1d Conv에서, Ngram의 역할 수행
        for kernel_size, filter_size in kernel_and_filter_sizes:
            layers.append(nn.Conv1d(in_channels, out_channels=filter_size, kernel_size=kernel_size, padding='same'))
            if conv_activation == "relu": layers.append(nn.ReLU(inplace=True))
            in_channels = filter_size
        self.convs = nn.Sequential(*layers)

        layers = []
        in_channels = hidden_size
        # 한번에 몇개씩 Pooling할지 설정
        for _, filter_size in kernel_and_filter_sizes:
            layers.append(nn.MaxPool1d(filter_size))
        self.pools = nn.Sequential(*layers)

        self.dropout1 = nn.Dropout(rate=dropout_rate)
        self.linear1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True)
        )
        self.dropout2 = nn.Dropout(rate=dropout_rate)
        self.linear2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        embeddings = self.embeddings(x)
        features = []
        for conv, pool in zip(self.convs, self.pools):
            x = conv(embeddings)
            x = pool(x)
            features.append(x)
        x = self.dropout1(torch.cat(features, dim = -1))

        x = self.linear1(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        return x


def string_to_example(encoding="UTF-8", max_length = 256, delete_prob = 0.15, add_prob = 0.5):
    """
    delete_prob: 띄어쓰기를 삭제할 확률
    add_prob: 띄어쓰기를 추가할 확률
    """
    pass

if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.training_config) as f:
        config = json.load(f)
    
    with open(args.char_file) as f:
        content = f.read()
        keys = ["<pad>", "<s>", "</s>", "<unk>"] + list(content)
        values = list(range(len(keys)))

        # Tensorflow의 textline dataset
        # 한 line씩 데이터셋으로 만들어주는 형태
        vocab_init = torch.tensor(keys, values, )