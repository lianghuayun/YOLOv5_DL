from torch import nn

class TimeSeriesModel(nn.Module):
    def __init__(self,input_size, hidden_size=32,
                 num_classes=1,num_layers=2,pad_idx=0):
        super(TimeSeriesModel,self).__init__()
        #self.rnn=nn.GRU(input_size,hidden_size,num_layers)
        self.rnn = nn.LSTM(input_size, hidden_size)
        #self.rnn=nn.RNN(input_size,hidden_size)
        self.fc=nn.Linear(hidden_size,num_classes)

    def forward(self,x):
        out, _ = self.rnn(x)
        out = self.fc(out[-1])
        return out
