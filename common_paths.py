__author__ = "ivallesp"

import os
import json

def _norm_path(path):
    """
    Decorator function intended for using it to normalize a the output of a path retrieval function. Useful for
    fixing the slash/backslash windows cases.
    """
    def normalize_path(*args, **kwargs):
        return os.path.normpath(path(*args, **kwargs))
    return normalize_path


def _assure_path_exists(path):
    """
    Decorator function intended for checking the existence of a the output of a path retrieval function. Useful for
    fixing the slash/backslash windows cases.
    """
    def assure_exists(*args, **kwargs):
        assert os.path.exists(path(*args, **kwargs))
        return path(*args, **kwargs)
    return assure_exists


def _is_output_path(path):
    """
    Decorator function intended for grouping the functions which are applied over the output of an output path retrieval
    function
    """
    @_norm_path
    @_assure_path_exists
    def check_existence_or_create_it(*args, **kwargs):
        if not os.path.exists(path(*args, **kwargs)):
            "Path didn't exist... creating it: {}".format(path(*args, **kwargs))
            os.makedirs(path(*args, **kwargs))
        return path(*args, **kwargs)
    return check_existence_or_create_it


def _is_input_path(path):
    """
    Decorator function intended for grouping the functions which are applied over the output of an input path retrieval
    function
    """
    @_norm_path
    @_assure_path_exists
    def check_existence(*args, **kwargs):
        return path(*args, **kwargs)
    return check_existence


@_is_input_path
def get_project_path():
    """
    Function used for retrieving the path where the project is located
    :return: the checked path (str|unicode)
    """
    with open("./settings.json") as f:
        settings = json.load(f)
    return settings["project_path"]


@_is_input_path
def get_data_path():
    """
    Function used to retrieve the path where the data will be stored
    :return: the checked path (str|unicode)
    """
    with open("./settings.json") as f:
        settings = json.load(f)
    return settings["data_path"]


@_is_input_path
def get_train_data_path():
    """
    Function used to get the path where the train data is stored
    :return: the checked path (str|unicode)
    """
    return os.path.join(get_data_path(), "images")


@_is_input_path
def get_test_data_path():
    """
    Function used to get the path where the test data is stored
    :return: the checked path (str|unicode)
    """
    return os.path.join(get_data_path(), "images_test")


