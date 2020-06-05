
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

## Typical execution


```
python -m modeling <data_type> <task_type> <config_path>
```

* data_type: "text" or "bin", if you have pickle already written to disk, choose "bin", otherwise choose "text"

* task_type:
   * "train": training one model for each of the research institution (generate as many models as there are institutions)
   * "train-all": training a model using all available data from all institutions
   * "test-self": test each model on the testing data from the resarch institution whose data was used for training
   * "test-cross": test each model on the testing data from all other research institutions
   * "test-all": TODO
   * "train-instances": training a model using a data that is splitted on the instance level
   * "test-instances": testing a model using a test set that was splitted on the instance level
   
Bootstrap should have been suppored by LightGBM, need to set the proper training parameter.
