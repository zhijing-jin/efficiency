# efficiency
This is a package of useful one-line logging functions made by Zhijing.

## Installation
Requirement: Python 3
```
pip install --upgrade git+git://github.com/zhijing-jin/efficiency.git
pip install --user -r requirements.txt
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
