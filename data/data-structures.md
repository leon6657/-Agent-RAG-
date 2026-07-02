# Python 数据结构

## 列表（List）

列表是可变的有序序列，可以包含不同类型的元素：

```python
fruits = ["苹果", "香蕉", "橘子"]
fruits.append("葡萄")
fruits[0]      # "苹果"
fruits[-1]     # "葡萄"
fruits[1:3]    # ["香蕉", "橘子"]
```

**列表推导式**：
```python
squares = [x**2 for x in range(10)]
even = [x for x in range(20) if x % 2 == 0]
```

## 元组（Tuple）

元组是不可变的有序序列，适合存储不应更改的数据：

```python
point = (3, 4)
x, y = point        # 解包
```

元组可以作为字典的键，列表不行。

## 字典（Dict）

字典是键值对的无序集合（Python 3.7+ 保持插入顺序）：

```python
student = {"name": "小明", "score": 95}
student["name"]          # "小明"
student.get("age", 0)    # 0（安全访问）
student.keys()           # 所有键
student.values()         # 所有值
```

## 集合（Set）

集合是无序的不重复元素集，支持数学集合运算：

```python
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}
a & b  # 交集 {3, 4}
a | b  # 并集 {1, 2, 3, 4, 5, 6}
a - b  # 差集 {1, 2}
```
