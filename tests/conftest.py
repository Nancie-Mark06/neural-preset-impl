"""让 `pytest` 从仓库根直接跑，不需要 `pip install -e .`。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
