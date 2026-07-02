# Python Decorator

A decorator is a function that takes another function and extends its behavior without explicitly modifying it.

## Example

```python
def my_decorator(func):
    def wrapper():
        print("Something is happening before the function is called.")
        func()
        print("Something is happening after the function is called.")
    return wrapper

@my_decorator
def say_whee():
    print("Whee!")
```

## Key Points

- Decorators wrap functions to add behavior
- They use the @ syntax
- The wrapper function can access the original function's arguments
