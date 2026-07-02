# Python 面向对象编程

## 类与对象

`class` 关键字定义类，`__init__` 是构造方法：

```python
class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def bark(self):
        return f"{self.name} says 汪汪！"

my_dog = Dog("小白", 3)
print(my_dog.bark())
```

## 继承

子类继承父类的属性和方法，可以覆写或扩展：

```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        pass

class Cat(Animal):
    def speak(self):
        return f"{self.name}: 喵~"

class Dog(Animal):
    def speak(self):
        return f"{self.name}: 汪汪！"
```

## 魔术方法

以双下划线开头和结尾的特殊方法：

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __repr__(self):
        return f"Vector({self.x}, {self.y})"

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
```

## 类方法与静态方法

```python
class MathUtils:
    @classmethod
    def from_string(cls, s):
        return cls(*map(int, s.split(",")))

    @staticmethod
    def is_positive(n):
        return n > 0
```

## 封装

- 单下划线 `_` 开头表示"保护"（约定）
- 双下划线 `__` 开头触发名称改写（name mangling）
