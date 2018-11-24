# efficiency
This is a package of useful one-line logging functions made by Zhijing.

## Installation
```
pip install --upgrade git+git://github.com/zhijing-jin/efficiency.git
pip install --user -r requirements.txt
```

## Logging Shortcuts
```python
>>> from efficiency.log import show_time, fwrite
>>> time_stamp = show_time()
⏰	time: 11241933-41
>>> time_stamp
'11241933' # means: Nov 24th, 19:33
>>> text = "This is handy!"
>>> fwrite(text, "temp.txt")

>>> num1 = 7
>>> num2 = 2
>>> num3 = 9
>>> show_var(["num1", "num2", "num3"])
num1 : 7
num2 : 2
num3 : 9
```

## Useful Functions
```python
>>> from efficiency.function import shell
>>> stdout, stderr = shell("cat temp.txt")
>>> stdout
b'This is handy!'
```
