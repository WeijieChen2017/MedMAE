import numpy as np

def dataset_division(dataset_list, train_ratio, val_ratio, test_ratio):
    """
    Divide the dataset into training, validation, and testing sets.
    :param dataset_list: list of dataset
    :param train_ratio: ratio of training set
    :param val_ratio: ratio of validation set
    :param test_ratio: ratio of testing set
    :return: train_list, val_list, test_list
    """
    selected_list = np.asarray(dataset_list)
    np.random.shuffle(selected_list)
    selected_list = list(selected_list)
    len_dataset = len(selected_list)
    selected_list = selected_list[:int(len_dataset * train_ratio)]

    val_list = selected_list[:int(len(selected_list)*val_ratio)]
    val_list.sort()
    test_list = selected_list[-int(len(selected_list)*test_ratio):]
    test_list.sort()
    train_list = list(set(selected_list) - set(val_list) - set(test_list))
    train_list.sort()

    return train_list, val_list, test_list