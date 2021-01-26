
import torch
import matplotlib.pyplot as plt
from load_data import load_data
import torch.utils.data as Data
from sklearn import preprocessing

torch.manual_seed(1)    # reproducible

# parameters
BATCH_SIZE = 128
EPOCH_NUM = 10

# load data
train_dataset, test_dataset = load_data()

# mini-batch
loader = Data.DataLoader(
    dataset=train_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # random shuffle for training
    num_workers=2,              # subprocesses for loading data
)

# define the model
net = torch.nn.Sequential(
                    torch.nn.Linear(28, 200),
                    torch.nn.ReLU(),
                    torch.nn.Linear(200, 3)
                    # torch.nn.ReLU(),
                    # torch.nn.Linear(100, 3),
)

# define the optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=0.02)

# define the loss
loss_func = torch.nn.CrossEntropyLoss()

if __name__ == '__main__':
    # train the model
    for epoch in range(EPOCH_NUM):
        for step, (batch_x, batch_y) in enumerate(loader):
            out = net(batch_x)
            loss = loss_func(out, batch_y)
            optimizer.zero_grad()  # clear gradients for next train
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
        print('epoch: {}, loss: {:.4}'.format(epoch, loss.data.item()))

    test_loader = Data.DataLoader(
        dataset=test_dataset,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=True,  # random shuffle for training
        num_workers=2,  # subprocesses for loading data
    )

    ## save the model
    PATH = 'D:/科研/碧月课题/代码/BPNN_Baseline/saved_model/model_GY02.pkl'
    torch.save(net, PATH)

    ## evaluate the model
    net.eval()
    eval_loss = 0
    eval_acc = 0
    for batch in test_loader:
        data, label = batch
        out = net(data)
        loss = loss_func(out, label)
        eval_loss += loss.data.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.item()
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(
        eval_loss / (len(test_dataset)),
        eval_acc / (len(test_dataset))
    ))

