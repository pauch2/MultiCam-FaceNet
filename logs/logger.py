import csv
import os
from datetime import datetime
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class AccessLogger:
    def __init__(self, log_file=config.LOG_PATH):
        self.log_file = log_file
        # Write header if file doesn't exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['timestamp', 'name', 'similarity', 'status'])

    def log(self, name, similarity, status):
        """
        Logs an access attempt.
        Status should be 'recognized' or 'unrecognized'
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, name, f"{similarity:.4f}", status])