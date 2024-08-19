#!/bin/bash

# 删除test.py文件
rm test_multiturn.py

# 从GitHub仓库下载test.py
curl -O https://raw.githubusercontent.com/THUlqs15/llm_test/main/test_multiturn.py

# 运行test.py
python test_multiturn.py
