# SafetyThickets


## Run

1. 生成配置 - 日志到 logs/generate_config/
```python -m src.st.generate_config```

2. 运行 expert - 日志到 logs/main/
```python -m src.st.main --expert_id 5```

3. 运行 base model - 日志到 logs/main/
```python -m src.st.main --expert_id base```