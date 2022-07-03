from GLOBALS import *
from impl_get_data import get_data_from_pickle
from impl_nets import Net


def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))


def get_fake_data():
    N = 100
    x, y = np.random.random((N, 60, 30)), np.random.random(N)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    N = 100
    x1 = np.sin(range(N)) + 0.01 * np.random.random(N)
    x2 = np.cos(range(N)) + 0.01 * np.random.random(N)
    y = 0.1 * np.power(x1, 2) + 0.5 * np.power(x2, 2)

    ax1.plot(x1)
    ax1.plot(x2)
    ax2.plot(y)
    plt.show()

    return x,  y


def plot_data(x):
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
        plt.pause(1)

    plt.close()


def run_simple_nn(x, y):
    # load a model
    net = Net()
    print(net)
    params = list(net.parameters())
    print(len(params))
    print(params[0].size())

    # train the model
    input = torch.randn(1, 1, 32, 32)
    out = net(input)
    print(out)
    net.zero_grad()
    out.backward(torch.randn(1, 10))

    output = net(input)
    target = torch.randn(10)  # a dummy target, for example
    target = target.view(1, -1)  # make it the same shape as output
    criterion = nn.MSELoss()

    loss = criterion(output, target)
    print(loss)

    print(loss.grad_fn)  # MSELoss
    print(loss.grad_fn.next_functions[0][0])  # Linear
    print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

    net.zero_grad()  # zeroes the gradient buffers of all parameters

    print('conv1.bias.grad before backward')
    print(net.conv1.bias.grad)

    loss.backward()

    print('conv1.bias.grad after backward')
    print(net.conv1.bias.grad)

    # create your optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    # in your training loop:
    optimizer.zero_grad()  # zero the gradient buffers
    output = net(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()  # Does the update

    # save the model
    pass

    # test the model
    pass

    # plot results
    pass


def main():
    x, y = get_fake_data()
    # x, y = get_data_from_pickle()
    # plot_data(x)
    # run_simple_nn(x, y)


if __name__ == '__main__':
    main()
