import matplotlib.pyplot as plt
import pickle

from GLOBALS import *


def get_data_from_pickle():
    with open('data/data_for_NN.pickle', 'rb') as file:
        data = pickle.load(file)

    x = [item[0] for item in data]
    y = [item[1] for item in data]

    return x, y


def get_stock_data():
    client = pymongo.MongoClient(os.environ['MONGODB_URL'])
    nt_data_db = client['nt_data']
    collection = nt_data_db['stocks']
    cursor = collection.find({}, {"_id": 0, "time": 1, "SPY": 1})
    return_dict = {'time': [], 'price': []}
    for item in cursor:
        return_dict['time'].append(item['time'])
        return_dict['price'].append(item['SPY']['adj'])
    return return_dict


def main():
    # get the .env value
    # print(os.environ['MONGODB_URL'])
    # return_value = get_stock_data()
    # plt.plot(return_value['price'])
    # plt.show()
    x, y = get_data_from_pickle()


if __name__ == '__main__':
    main()
