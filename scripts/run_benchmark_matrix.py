#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Proxy Runner for 84 Benchmark Matrix (Table 4, ICML 2026) located in scripts/.
Executes system/run_benchmark_matrix.py with working directory set to system/.
"""

import os
import sys
import subprocess

if __name__ == "__main__":
    system_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "system"))
    target_script = os.path.join(system_dir, "run_benchmark_matrix.py")
    
    cmd = [sys.executable, target_script] + sys.argv[1:]
    sys.exit(subprocess.call(cmd, cwd=system_dir))
