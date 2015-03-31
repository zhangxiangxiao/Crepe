# Data Tools for Crepe

This directory contains data tools for Crepe. Below is the documentation.

## `csv2t7b.lua`

This is a tool to convert the csv [datasets distribution](http://goo.gl/JyCnZq) for our article "Text Understanding from Scratch" to a binary format that can be used in Crepe's training program. We called this binary format `t7b`, which stands for torch-7-binary.

### Usage

The program contains 2 command-line parameters that you can use:
* `-input [file]`: Input csv file.
* `-output [file]`: Output t7b file.

The input file is a csv format where the first column are numbers indicating class labels, starting from 1. The rest of the columns are considered text fields. These fields follow standard csv escaping rules such as one can use double quote `"` to quote the field, and use 2 double quotes `""` to escape for a double quote character. New lines are escaped by a backslash then following a character `n`, i.e., `\n`.

The output file is a binary serialization of data in torch 7. The postfix `t7b` means "torch-7-binary". These generated files can be direct used as training or validating datasets for the Crepe training program component. If you want to know the details of this format, take a look at the following section.

### More on `t7b` format

First of all, the postfix name `t7b` is not only used as a postfix for dataset files. Instead, I used this postfix to any data that is serialized to binary format by Torch 7. For example, the Crepe trainig program also generates files ending in `t7b` as model serialization or checkpoints (`sequential_EPOCHES_TIME.t7b` or `main_EPOCHES_TIME.t7b`).

For the datasets in `t7b` format, you can load them using standard torch calls. For example
```
> train = torch.load("train.t7b")
```

Then, the variable `train` contains 3 members. They are:
* `train.content`: a `torch.ByteTensor` that stores concatenated string data. Each string is ended with `NULL`(0).
* `train.index`: a lua table for which `train.index[i]` is a 2-D `torch.LongTensor`. `train.index[i][j][k]` indicates the offset in `train.content` for the string in class i, j-th sample and k-th field (remember that there can be multiple text fields in csv format).
* `train.length`: a lua table for which `train.length[i]` is a 2-D `torch.LongTensor`. `train.length[i][j][k]` indicates the length for the string in class i, j-th sample and k-th field. The length does not count the ending `NULL`.

To use this format, one can use the `ffi.string` call [provided by luajit](http://luajit.org/ext_ffi_api.html) to return a lua string. For example
```
> ffi = require("ffi")
> str = ffi.string(torch.data(train.content:narrow(1, train.index[3][8][2], 1)))
```

Then, variable `str` is a lua string representing the string of class 3, 8th sample and 2nd field. By the definitions above, `str:len()` should be equal to `train.length[3][8][2]`.

