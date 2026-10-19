"""Compatibility entry point; the current Aligned queue lives in aligned_queue.py."""
import runpy
from pathlib import Path
if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).with_name('aligned_queue.py')), run_name='__main__')
