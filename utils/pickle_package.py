import pickle5 as pickle


def object_to_pickle(instant, address):
    with open(address, "wb") as f:
        pickle.dump(instant, f)


def pickle_to_object(address):
    with open(address, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    instant = 5
    object_to_pickle(instant,"./test.pickle")
    print(pickle_to_object("./test.pickle"))