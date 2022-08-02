from GLOBALS import *
from impl_nets import LSTMNet
from impl_get_data import get_data_from_pickle


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
        curr_x += x[i]
        x[i] = curr_x
        curr_y = np.sum(curr_x[:, 29:])
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


def get_tensors(X, y):
    y_tensors = []
    X_tensors_final = []
    for i in range(X.shape[0]):
        x_item = X[i]
        x_item_transposed = x_item.T
        x_item_tensor = Variable(torch.Tensor(x_item_transposed))
        x_item_tensor_final = torch.reshape(x_item_tensor, (x_item_tensor.shape[0], 1, x_item_tensor.shape[1]))
        X_tensors_final.append(x_item_tensor_final)
        y_tensors.append(Variable(torch.Tensor([[y[i]]])))
    return X_tensors_final, y_tensors


def preprocessing(x, y, train_percentage=0.7):
    x = np.array(x)
    y = np.array(y)
    # first * for training
    to_train = int(train_percentage * len(x))
    to_validate = to_train + int((len(x) - to_train)*0.6)

    X_train = x[:to_train, :]
    y_train = y[:to_train]

    X_val = x[to_train:to_validate, :]
    y_val = y[to_train:to_validate]

    X_test = x[to_validate:, :]
    y_test = y[to_validate:]

    X_train_tensors_final, y_train_tensors = get_tensors(X_train, y_train)
    X_val_tensors_final, y_val_tensors = get_tensors(X_val, y_val)
    X_test_tensors_final, y_test_tensors = get_tensors(X_test, y_test)

    return X_train_tensors_final, y_train_tensors, X_val_tensors_final, y_val_tensors, X_test_tensors_final, y_test_tensors


def run_simple_nn(x, y, num_epochs, learning_rate, input_size, hidden_size, num_layers, fc_layer_size, num_classes, train_percentage=0.7):
    """
    Preprocessing
    """
    X_train_tensors_final, y_train_tensors, X_val_tensors_final,\
    y_val_tensors, X_test_tensors_final, y_test_tensors = preprocessing(x, y, train_percentage=train_percentage)

    """
    Load the model
    """

    seq_length = X_train_tensors_final[0].shape[0]
    net = LSTMNet(
        num_classes=num_classes,
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        seq_length=seq_length,
        fc_layer_size=fc_layer_size
    )
    criterion = torch.nn.MSELoss()  # mean-squared error for regression
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    print(net)

    """
    Train the model
    """
    losses_train = []
    losses_val = []
    for epoch in range(num_epochs):

        y_hat = []
        y_real = []
        losses = []
        for i, i_batch in enumerate(X_train_tensors_final):
            outputs = net(i_batch)  # forward pass
            optimizer.zero_grad()  # calculate the gradient, manually setting to 0

            # obtain the loss function
            y_train_tensor = y_train_tensors[i]
            loss = criterion(outputs, y_train_tensor)

            loss.backward()  # calculates the loss of the loss function

            optimizer.step()  # improve from loss, i.e backprop

            y_hat.append(outputs.item())
            y_real.append(y_train_tensor.item())
            losses.append(loss.item())
            print(f"\rEpoch-batch: {epoch}-{i}, loss:{loss.item() : 1.5f}", end='')
        print(f"\nEpoch: {epoch}, training loss:{np.mean(losses) : 1.5f}")


        # val step
        losses_train.append(np.mean(losses))

        net.eval()
        losses_val_values = []
        for i, i_batch in enumerate(X_val_tensors_final):
            outputs = net(i_batch)  # forward pass
            y_train_tensor = y_train_tensors[i]
            loss = criterion(outputs, y_train_tensor)
            losses_val_values.append(loss.item())
        losses_val.append(np.mean(losses_val_values))

        print(f"\rEpoch: {epoch}, valuation loss:{np.mean(losses_val) : 1.5f}\n")

        # plt.cla()
        # plt.clf()
        # plt.plot(losses_train, label='losses_train')
        # plt.plot(losses_val, label='losses_val')
        # plt.legend()
        # plt.pause(0.01)

        # print(f"Epoch: {epoch}, loss:{np.mean(losses) : 1.5f}")
        # plt.plot(y_real, label='real')
        # plt.plot(y_hat, label='predicted')
        # plt.legend()
        # plt.show()

    # save the model
    torch.save(net.state_dict(), PATH)
    return losses_train, losses_val , y_real, y_hat


def test_model(x, y, input_size, hidden_size, num_layers, fc_layer_size, num_classes, train_percentage=0.7):
    """
    Preprocessing
    """
    X_train_tensors_final, y_train_tensors, X_val_tensors_final, y_val_tensors, X_test_tensors_final, y_test_tensors = preprocessing(x, y)

    """ 
    Test the model
    """

    seq_length = X_train_tensors_final[0].shape[0]

    net = LSTMNet(
        num_classes=num_classes,
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        seq_length=seq_length,
        fc_layer_size=fc_layer_size
    )
    net.load_state_dict(torch.load(PATH))
    net.eval()

    y_hat = []
    y_real = []
    for i, i_batch in enumerate(X_test_tensors_final):
        outputs = net(i_batch)
        y_hat.append(outputs.item())
        y_real.append(y_test_tensors[i].item())

    # # plot results
    # plt.plot(y_real, label='real')
    # plt.plot(y_hat, label='predicted')
    # plt.legend()
    # plt.show()

    return y_hat, y_real


def grid_search(x, y):
    num_epochss = [300]
    learning_rates=[0.001]
    hidden_sizes=[8,64]
    fc_layer_sizes=[8, 64]
    input_size = 60
    num_layers = 1
    num_classes = 1

    result = pd.DataFrame(columns=['num_epochs', 'learning_rates', 'input_size', 'hidden_size', 'fc_layer_size',
                             'losses_train', 'losses_val', 'train_y_real', 'train_y_hat', 'test_y_hat', 'test_y_real'])
    for num_epochs in num_epochss:
        for learning_rate in learning_rates:
            for hidden_size in hidden_sizes:
                for fc_layer_size in fc_layer_sizes:
                    losses_train, losses_val, tr_y_real, tr_y_hat = run_simple_nn(x, y, num_epochs=num_epochs,
                                                                    learning_rate=learning_rate,
                                                                    input_size=input_size,
                                                                    hidden_size=hidden_size,
                                                                    num_layers=num_layers,
                                                                    fc_layer_size=fc_layer_size,
                                                                    num_classes=num_classes)
                    y_hat, y_real = test_model(x, y,input_size=input_size,
                                                    hidden_size=hidden_size,
                                                    num_layers=num_layers,
                                                    fc_layer_size=fc_layer_size,
                                                    num_classes=num_classes)
                    result = result.append(pd.DataFrame([[num_epochs, learning_rate, input_size, hidden_size, fc_layer_size,
                                                          losses_train, losses_val, tr_y_real, tr_y_hat, y_hat, y_real]],
                                                          columns=result.columns))

    with open('grid_search_results3.pickle', 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_grid_search_results():
    with open('grid_search_results3.pickle', 'rb') as handle:
        result = pickle.load(handle)
    return result


def main():
    x, y = get_fake_data()
    # x, y = get_data_from_pickle()
    # plot_data_x(x)
    # plot_data_y(y)

    num_epochs = 2  # 1000 epochs
    learning_rate = 0.001  # 0.001 lr
    input_size = 60  # number of features
    hidden_size = 4  # number of features in hidden state
    num_layers = 1  # number of stacked lstm layers
    fc_layer_size = 16
    num_classes = 1  # number of output classes

    # run_simple_nn(x, y, num_epochs=num_epochs,
    #                     learning_rate=learning_rate,
    #                     input_size=input_size,
    #                     hidden_size=hidden_size,
    #                     num_layers=num_layers,
    #                     fc_layer_size=fc_layer_size,
    #                     num_classes=num_classes)

    # test_model(x, y,input_size=input_size,
    #                 hidden_size=hidden_size,
    #                 num_layers=num_layers,
    #                 fc_layer_size=fc_layer_size,
    #                 num_classes=num_classes)

    grid_search(x, y)
    # print(load_grid_search_results())


if __name__ == '__main__':
    PATH = 'models/sample_net.nn'

    main()
