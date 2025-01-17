from GLOBALS import *


class LSTMNet(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length, fc_layer_size):
        super(LSTMNet, self).__init__()
        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length
        self.fc_layer_size = fc_layer_size

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers)  # lstm
        self.fc_1 = nn.Linear(hidden_size, fc_layer_size)  # fully connected 1
        self.fc = nn.Linear(fc_layer_size, num_classes)  # fully connected last layer

        # self.fc_2 = nn.Linear(5, 5)  # fully connected 2

        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(1), self.hidden_size))  # hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(1), self.hidden_size))  # internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size)  # reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out)  # first Dense
        out = self.relu(out)  # relu
        out = self.fc(out)    # Final Output

        return out
