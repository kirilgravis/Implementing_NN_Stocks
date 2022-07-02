from GLOBALS import *
from impl_get_data import get_data_from_pickle
from impl_nets import Net


def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))


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
    pass

    # save the model
    pass

    # test the model
    pass

    # plot results
    pass


def main():
    x, y = get_data_from_pickle()
    # plot_data(x)
    run_simple_nn(x, y)


if __name__ == '__main__':
    main()
