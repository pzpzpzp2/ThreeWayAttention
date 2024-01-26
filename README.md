following google and chatgpt, the way to build a wheel is 

```python3 -m pip install --upgrade build```
```python3 -m build```

then add 

```threewayattentionpackage = {path = "./models/archs/threewayattention/threewayattentionpackage-0.1.0-py3-none-any.whl"}```

to your

```[tool.poetry.dependencies]```