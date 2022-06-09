import csv

import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim, tensor


class SentimentNet(nn.Module):
    def __init__(self, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        super(SentimentNet, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = 1
        print(x.size())
        x = x.long()
        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.sigmoid(out)

        out = out.view(batch_size, -1)
        out = out[:, -1]
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden

def evaluate_epoch(
        train_x, train_y,
        val_x, val_y,
        test_x, test_y,
        model,
        criterion
):
    """Evaluate the `model` on the train and validation set."""

    def _get_metrics(data_x, data_y):
        y_true, y_pred, y_score = [], [], []
        correct, total = 0, 0
        running_loss = []
        for i in range(len(data_y)):
            X = data_x[i]
            y = data_y[i]
            with torch.no_grad():
                output = model(X)
                predicted = torch.argmax(output.data)
                y_true.append(y)
                y_pred.append(predicted)
                y_score.append(softmax(output.data)[1])
                total += 1
                correct += (predicted == y).sum().item()
                running_loss.append(criterion(output, y).item())
        y_true = tensor(y_true)
        y_pred = tensor(y_pred)
        y_score = tensor(y_score)
        loss = np.mean(running_loss)
        acc = correct / total
        auroc = metrics.roc_auc_score(y_true, y_score)
        return acc, loss, auroc

    train_acc, train_loss, train_auc = _get_metrics(train_x, train_y)
    val_acc, val_loss, val_auc = _get_metrics(val_x, val_y)
    test_acc, test_loss, test_auc = _get_metrics(test_x, test_y)

    stats_at_epoch = [
        val_acc,
        val_loss,
        val_auc,
        train_acc,
        train_loss,
        train_auc,
        test_acc, test_loss, test_auc
    ]
    print(stats_at_epoch)
    return val_loss


ET_data = None
ENT_data = None
x = []
y = []

with open(r"spiketrain.csv") as c:
    r = csv.reader(c)
    for data in r:
        animal_class = data[1]
        # print(animal_class)
        train = data[2:]
        train = [int(i) for i in train]
        if animal_class == 'ET':
            # if ET_data == None:
            #     ET_data = torch.tensor(train).unsqueeze(0)
            # else:
            #     ET_data = torch.cat((ET_data, torch.tensor(train).unsqueeze(0)), 0)
            x.append(train)
            y.append(1)
        elif animal_class == 'ENT':
            # if ENT_data == None:
            #     ENT_data = torch.tensor(train).unsqueeze(0)
            # else:
            #     ENT_data = torch.cat((ENT_data, torch.tensor(train).unsqueeze(0)), 0)
            x.append(train)
            y.append(0)
    print(len(x))
    print(len(y))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.25)
x_train = tensor(x_train, dtype=torch.float)
y_train = tensor(y_train)
x_val = tensor(x_val, dtype=torch.float)
y_val = tensor(y_val)
x_test = tensor(x_test, dtype=torch.float)
y_test = tensor(y_test)

output_size = 1
embedding_dim = 1
hidden_dim = 512
n_layers = 2

model = SentimentNet(output_size, embedding_dim, hidden_dim, n_layers)

lr=0.005
criterion = nn.CrossEntropyLoss()

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use RMSprop; the optim package contains many other
# optimization algorithms. The first argument to the RMSprop constructor tells the
# optimizer which Tensors it should update.
learning_rate = 1e-3
patience = 5
global_min_loss = None
curr_count_to_patience = 0
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
count = 0
for t in range(200):
    for i in range(len(y_train)):
        count += 1
        h = model.init_hidden(1)
        if curr_count_to_patience == patience:
            break
        # Forward pass: compute predicted y by passing x to the model.
        y_pred, h = model(x_train[i], h)

        # Compute and print loss.
        loss = criterion(y_pred, y_train[i])
        if count % 100 == 99:
            print("Epoch", count, loss.item())
        current_loss = evaluate_epoch(x_train, y_train, x_val, y_val, x_test, y_test, model, criterion)
        if (global_min_loss == None or current_loss < global_min_loss):
            global_min_loss = current_loss
            curr_count_to_patience = 0
        else:
            curr_count_to_patience += 1

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model). This is because by default, gradients are
        # accumulated in buffers( i.e, not overwritten) whenever .backward()
        # is called. Checkout docs of torch.autograd.backward for more details.
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()
