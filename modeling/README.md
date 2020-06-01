
`__init__.py`: for python path

`__main__.py`: the entrance of the program so that we can call it from commandline. It also parse commandline arguments.

`booster.py`: actual code to call LightGBM

`common.py`:	common functions to do various tasks

`load_data.py`: loading the text/binary(pickle) training and testing data.
If the input format is text, it will also write pickle files so that next time the data loading would be faster.

`test.py`: template code to be called by "__main__.py" proper functions for testing. It outputs a pickle file that
contains scores in addition to some meta information about examples, e.g. cruise ID, longitute, latitude

`train.py`: template code to be called by "__main__.py" proper functions for training.

`config.json`: config such as the input data path, and the directory to write the models
