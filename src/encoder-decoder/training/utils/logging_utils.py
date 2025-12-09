"""Logging utilities"""

import sys
import re
from pathlib import Path


class Tee:
    """Tee stdout to a log file safely."""
    
    # ANSI escape code pattern
    ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    
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
                # Strip ANSI codes before writing to file
                clean_s = self.ANSI_ESCAPE.sub('', s)
                self.file.write(clean_s)
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
