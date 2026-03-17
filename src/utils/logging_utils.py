import logging
import os
import sys
import time

def setup_logging(
    task_name, # train, eval, etc.
    log_level=logging.INFO,
    log_dir="logs",
    run_id=None,
):

    # Create logs directory if it doesn't exist
    log_dir = os.path.join(log_dir, task_name)
    os.makedirs(log_dir, exist_ok=True)

    # Create log file path with timestamp
    time_stamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(log_dir, f"{time_stamp}_{run_id}.log") if run_id else os.path.join(log_dir, f"{time_stamp}.log")

    # Setup logging format
    log_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    date_format = "%m/%d/%Y %H:%M:%S"
    
    # Clear any existing handlers to avoid conflicts
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Setup logging to both file and console with force=True to override existing config
    logging.basicConfig(
        format=log_format,
        datefmt=date_format,
        level=log_level,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ],
        force=True  # This forces reconfiguration even if basicConfig was called before
    )