"""
Development entry point.

Run from the project root:
    uv run python main.py
    # or
    uv run uvicorn backend.main:app --reload
"""

import sys
import os

# Put the backend package on the path so `uvicorn backend.main:app` also works
# when invoked directly from the project root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "main:app",           # module:attribute inside backend/
        app_dir="backend",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["backend"],
    )
