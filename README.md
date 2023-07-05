# efficiency

[![Pypi](https://img.shields.io/pypi/v/efficiency.svg)](https://pypi.org/project/efficiency)
[![Downloads](https://pepy.tech/badge/efficiency)](https://pepy.tech/project/efficiency)
[![Downloads](https://pepy.tech/badge/efficiency/month)](https://pepy.tech/project/efficiency/month)
[![MIT_License](https://camo.githubusercontent.com/890acbdcb87868b382af9a4b1fac507b9659d9bf/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f6c6963656e73652d4d49542d626c75652e737667)](LICENCE)

This is a package of useful one-line logging functions made by Zhijing.

## Installation

Requirement: Python 3

```
pip install --upgrade  git+https://github.com/zhijing-jin/efficiency.git
```

## Logging Shortcuts

Obtain time:

```python
>>> from efficiency.log import show_time, fwrite
>>> time_stamp = show_time()
	time: 11241933-41
>>> time_stamp
'11241933' # means: Nov 24th, 19:33
```

Writing out to files by one line:

```python
>>> text = "This is handy!"
>>> fwrite(text, "temp.txt")
```

Printing out variables (name + value) easily:

```python
>>> num1 = 7
>>> num2 = 2
>>> num3 = 9
>>> show_var(["num1", "num2", "num3"])
num1 : 7
num2 : 2
num3 : 9
```

### ML-Related

```python
>>> from efficiency.log import gpu_mem
>>> gpu_mem(gpu_id=0)
4101 # Currently, GPU Memory of GPU #0 is 4101 MiB
>>> from efficiency.function import set_seed
>>> set_seed(0)
[Info] seed set to: 0 # set the seed for random, numpy and pytorch
```

## Useful Functions

```python
>>> from efficiency.function import shell
>>> stdout, stderr = shell("cat temp.txt")
>>> stdout
b'This is handy!'
```

# Miscellaneous Functions

#### Formatting README.md Better

This is the way to automatically generate table of contents for an .md file using [markdown_toc](https://github.com/alexharv074/markdown_toc):

```
# Step 1. download the file to your local execution path `/usr/local/bin/`
curl \
  https://raw.githubusercontent.com/alexharv074/markdown_toc/master/mdtoc.rb -o \
  /usr/local/bin/mdtoc.rb

# Step 2. generate the ToC in the "copy" mode
mdtoc.rb README.md

# Alternatively, you can automatically add the content to your copyboard and paste it to your .md
mdtoc.rb README.md | pbcopy
```

