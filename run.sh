#!/bin/bash

# 删除test.py文件
rm test1.py

# 从GitHub仓库下载test.py
curl -O https://raw.githubusercontent.com/THUlqs15/llm_test/main/test1.py

# 运行test.py
python test1.py
