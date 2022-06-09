import csv
import pickle

import torch
from sklearn.model_selection import train_test_split
from torch import nn, tensor
import numpy as np
from torch.nn.functional import softmax
from sklearn import metrics


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

model = torch.nn.Sequential(
    nn.Linear(3080, 512),
    nn.ReLU(),
    nn.Linear(512, 2)
)
loss_fn = torch.nn.CrossEntropyLoss()

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use RMSprop; the optim package contains many other
# optimization algorithms. The first argument to the RMSprop constructor tells the
# optimizer which Tensors it should update.
learning_rate = 1e-3
patience = 5
global_min_loss = None
curr_count_to_patience = 0
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
for t in range(200):
    if curr_count_to_patience == patience:
        break
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x_train)

    # Compute and print loss.
    loss = loss_fn(y_pred, y_train)
    if t % 100 == 99:
        print("Epoch", t, loss.item())
    current_loss = evaluate_epoch(x_train, y_train, x_val, y_val, x_test, y_test, model, loss_fn)
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

filename = 'NN_model.sav'
pickle.dump(model, open(filename, 'wb'))

print(model[0].weight.grad)
print(model[2].weight.grad)