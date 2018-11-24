# efficiency
This is a package of useful one-line logging functions made by Zhijing.

## Installation
```
pip install --upgrade git+git://github.com/zhijing-jin/efficiency.git
pip install --user -r requirements.txt
```

## Inside Python
```
>>> from efficiency import log
>>> time_stamp = log.show_time()
⏰	time: 11241933-41
>>> time_stamp
'11241933' # means: Nov 24th, 19:33
>>> text = "This is handy!"
>>> log.fwrite(text, "temp.txt")
```


