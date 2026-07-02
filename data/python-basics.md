# Python 函数基础

## 定义函数

使用 `def` 关键字定义函数：

```python
def greet(name):
    """向指定的人打招呼"""
    return f"你好，{name}！"
```

## 参数类型

### 位置参数
按顺序传入，必须提供：
```python
def add(a, b):
    return a + b

add(3, 5)  # 8
```

### 默认参数
参数可以有默认值：
```python
def power(base, exp=2):
    return base ** exp

power(3)     # 9
power(3, 3)  # 27
```

### 可变参数
`*args` 接收任意数量的位置参数，`**kwargs` 接收关键字参数：
```python
def sum_all(*args):
    return sum(args)

def print_info(**kwargs):
    for k, v in kwargs.items():
        print(f"{k}: {v}")
```

## 返回值

- 函数可以返回多个值（实际上是一个元组）
- 没有 `return` 语句时返回 `None`

```python
def min_max(lst):
    return min(lst), max(lst)

low, high = min_max([3, 1, 4, 1, 5, 9])
```

## 作用域

- 函数内部变量是局部作用域
- `global` 关键字可以声明全局变量
- `nonlocal` 关键字用于嵌套函数

```python
count = 0

def increment():
    global count
    count += 1
```
