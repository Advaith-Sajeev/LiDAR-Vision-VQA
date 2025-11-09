"""Logging utilities"""

import sys
from pathlib import Path


class Tee:
    """Tee stdout to a log file safely."""
    
    def __init__(self, logfile: Path):
        logfile.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(logfile, "a", buffering=1)
        self.stdout = sys.stdout
        self.closed = False
        
    def write(self, s):
        try:
            self.stdout.write(s)
        except Exception:
            pass
        if not self.closed:
            try:
                self.file.write(s)
            except Exception:
                pass
                
    def flush(self):
        try:
            self.stdout.flush()
        except Exception:
            pass
        if not self.closed:
            try:
                self.file.flush()
            except Exception:
                pass
                
    def close(self):
        if self.closed:
            return
        try:
            self.file.flush()
            self.file.close()
        except Exception:
            pass
        self.closed = True
