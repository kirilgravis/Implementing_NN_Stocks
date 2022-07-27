import numpy as np

from GLOBALS import *
from impl_get_data import get_data_from_pickle
from impl_nets import LSTMNet


def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))


def get_fake_data():
    N = 100
    x, y = np.random.random((N, 60, 30)), np.random.random(N)

    # create "stocks" series
    for i in range(N):
        curr_x = np.ones((60, 30))
        curr_x = np.cumsum(curr_x, 1)
        curr_x = np.sin(curr_x)
        curr_x = curr_x + x[i] * 0.3
        x[i] = curr_x
        curr_y = np.sum(curr_x[:, 20:]) / 60
        y[i] = curr_y

    return x,  y


def plot_data_x(x):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax = plt.axes(projection='3d')
    for x_item in x:
        ax.clear()

        # 3D plot
        X = list(range(x_item.shape[1]))
        Y = list(range(x_item.shape[0]))
        X, Y = np.meshgrid(X, Y)
        ax.plot_surface(X, Y, x_item, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        ax.set_title('surface')

        # 2D plot
        # ax.matshow(x_item)

        plt.tight_layout()
        plt.pause(0.01)

    plt.close()


def plot_data_y(y):
    plt.plot(y)
    plt.show()


def run_simple_nn(x, y, train_percentage=0.8):
    """
    Preprocessing
    """
    x = np.array(x)
    y = np.array(y)
    # first * for training
    to_train = int(train_percentage * len(x))

    X_train = x[:to_train, :]
    y_train = y[:to_train]

    X_test = x[to_train:, :]
    y_test = y[to_train:]

    y_train_tensors = []
    X_train_tensors_final = []
    for i in range(X_train.shape[0]):
        x_item = X_train[i]
        x_item_transposed = x_item.T
        x_item_tensor = Variable(torch.Tensor(x_item_transposed))
        x_item_tensor_final = torch.reshape(x_item_tensor, (x_item_tensor.shape[0], 1, x_item_tensor.shape[1]))
        X_train_tensors_final.append(x_item_tensor_final)
        y_train_tensors.append(Variable(torch.Tensor([[y_train[i]]])))

    y_test_tensors = []
    X_test_tensors_final = []
    for i in range(X_test.shape[0]):
        x_item = X_test[i]
        x_item_transposed = x_item.T
        x_item_tensor = Variable(torch.Tensor(x_item_transposed))
        x_item_tensor_final = torch.reshape(x_item_tensor, (x_item_tensor.shape[0], 1, x_item_tensor.shape[1]))
        X_test_tensors_final.append(x_item_tensor_final)
        y_test_tensors.append(Variable(torch.Tensor([[y_test[i]]])))

    """
    Load the model
    """
    num_epochs = 10  # 1000 epochs
    learning_rate = 0.001  # 0.001 lr

    input_size = 60  # number of features
    hidden_size = 30  # number of features in hidden state
    num_layers = 1  # number of stacked lstm layers

    num_classes = 1  # number of output classes
    seq_length = X_train_tensors_final[0].shape[0]
    net = LSTMNet(
        num_classes=num_classes,
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        seq_length=seq_length
    )
    criterion = torch.nn.MSELoss()  # mean-squared error for regression
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    print(net)

    """
    Train the model
    """
    for epoch in range(num_epochs):

        y_hat = []
        losses = []
        for i, i_batch in enumerate(X_train_tensors_final):
            outputs = net(i_batch)  # forward pass
            optimizer.zero_grad()  # caluclate the gradient, manually setting to 0

            # obtain the loss function
            y_train_tensor = y_train_tensors[i]
            loss = criterion(outputs, y_train_tensor)

            loss.backward()  # calculates the loss of the loss function

            optimizer.step()  # improve from loss, i.e backprop

            y_hat.append(outputs.item())
            losses.append(loss.item())
            print(f"\rEpoch-batch: {epoch}-{i}, loss:{loss.item() : 1.5f}", end='')
        print(f"\nEpoch: {epoch}, loss:{np.mean(losses) : 1.5f}")

        # if epoch % 10 == 0:
        #     # print(f"Epoch: {epoch}, loss:{np.mean(losses) : 1.5f}")
        #     plt.plot(y_train, label='real')
        #     plt.plot(y_hat, label='predicted')
        #     plt.legend()
        #     plt.show()

    # save the model
    torch.save(net.state_dict(), PATH)

    # test model
    test_model(x, y, train_percentage)


def test_model(x, y, train_percentage=0.8):
    """
        Preprocessing
        """
    x = np.array(x)
    y = np.array(y)
    # first * for training
    to_train = int(train_percentage * len(x))

    X_train = x[:to_train, :]
    y_train = y[:to_train]

    X_test = x[to_train:, :]
    y_test = y[to_train:]

    y_train_tensors = []
    X_train_tensors_final = []
    for i in range(X_train.shape[0]):
        x_item = X_train[i]
        x_item_transposed = x_item.T
        x_item_tensor = Variable(torch.Tensor(x_item_transposed))
        x_item_tensor_final = torch.reshape(x_item_tensor, (x_item_tensor.shape[0], 1, x_item_tensor.shape[1]))
        X_train_tensors_final.append(x_item_tensor_final)
        y_train_tensors.append(Variable(torch.Tensor([[y_train[i]]])))

    y_test_tensors = []
    X_test_tensors_final = []
    for i in range(X_test.shape[0]):
        x_item = X_test[i]
        x_item_transposed = x_item.T
        x_item_tensor = Variable(torch.Tensor(x_item_transposed))
        x_item_tensor_final = torch.reshape(x_item_tensor, (x_item_tensor.shape[0], 1, x_item_tensor.shape[1]))
        X_test_tensors_final.append(x_item_tensor_final)
        y_test_tensors.append(Variable(torch.Tensor([[y_test[i]]])))

    input_size = 60  # number of features
    hidden_size = 30  # number of features in hidden state
    num_layers = 1  # number of stacked lstm layers

    num_classes = 1  # number of output classes

    seq_length = X_train_tensors_final[0].shape[0]

    # test the model
    net = LSTMNet(
        num_classes=num_classes,
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        seq_length=seq_length
    )
    net.load_state_dict(torch.load(PATH))
    net.eval()

    y_hat = []
    for i, i_batch in enumerate(X_test_tensors_final):
        outputs = net(i_batch)
        y_hat.append(outputs.item())

    # plot results
    plt.plot(y_test, label='real')
    plt.plot(y_hat, label='predicted')
    plt.legend()
    plt.show()


def main():
    # x, y = get_fake_data()
    x, y = get_data_from_pickle()
    # plot_data_x(x)
    # plot_data_y(y)
    # run_simple_nn(x, y)
    test_model(x, y)


if __name__ == '__main__':
    PATH = 'models/sample_net.nn'

    main()
