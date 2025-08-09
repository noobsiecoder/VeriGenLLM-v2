"""
Code to log info

This module provides a centralized logging system for the VeriGen LLM project.
It creates both file and console logs with consistent formatting across all
components of the evaluation framework.

Author: Abhishek Sriram <noobsiecoder@gmail.com>
Date:   Aug 1st, 2025

Note: LLM was used to generate comments
"""

import logging
from datetime import datetime
from pathlib import Path


class Logger:
    """
    Logger class

    A custom logger implementation that provides:
    - Automatic log file creation with daily rotation
    - Consistent formatting across the project
    - Both file and console output
    - Prevention of duplicate handlers

    This ensures all components (LLM clients, evaluation scripts, etc.)
    have consistent logging behavior.
    """

    def __init__(self, name: str, level=logging.INFO):
        """
        Initialize a logger instance with file and console handlers

        Parameters:
        -----------
        name : str
            Logger name, typically the module or component name
            (e.g., 'claude_client', 'pass-k_metrics', 'local_llm')
            This name appears in log messages and determines the log filename
        level : int, default=logging.INFO
            Minimum logging level to capture
            - DEBUG: Detailed information for diagnosing problems
            - INFO: General informational messages
            - WARNING: Warning messages
            - ERROR: Error messages
            - CRITICAL: Critical problems

        Notes:
        ------
        Log files are created in the 'logs' directory at the project root
        with the format: {name}_{YYYY-MM-DD}.log
        """
        # Create or get a logger with the specified name
        # Using getLogger ensures we reuse existing loggers with the same name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Check if handlers already exist to prevent duplicates
        # This is important when the same logger is initialized multiple times
        if not self.logger.handlers:
            # Define log directory path relative to this file
            # __file__ is the current file (logger.py)
            # parents[2] goes up two levels to reach project root
            # Assumes structure: project_root/verigenllm_v2/utils/logger.py
            parent_dir = Path(__file__).resolve().parents[2]
            log_dir = parent_dir / "logs"

            # Create logs directory if it doesn't exist
            # parents=True creates intermediate directories if needed
            # exist_ok=True prevents error if directory already exists
            log_dir.mkdir(parents=True, exist_ok=True)

            # Define log file name with timestamp for daily rotation
            # This creates separate log files for each day, making it easier to:
            # - Track issues by date
            # - Manage log file sizes
            # - Archive old logs
            log_file = log_dir / f"{name}_{datetime.now().strftime('%Y-%m-%d')}.log"

            # File handler - writes log messages to file
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)

            # Console handler - displays log messages in terminal
            # Useful for real-time monitoring during development/debugging
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)

            # Define log message format
            # Format: 2024-01-20 10:30:45,123 - module_name - INFO - Log message here
            # Components:
            # - asctime: Timestamp with milliseconds
            # - name: Logger name (module/component)
            # - levelname: Log level (INFO, ERROR, etc.)
            # - message: The actual log message
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

            # Apply formatter to both handlers
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            # Add handlers to logger
            # Both handlers will process each log message
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def get_logger(self):
        """
        Return the configured logger instance

        Returns:
        --------
        logging.Logger
            The configured logger ready for use

        Example:
        --------
        >>> log = Logger("my_module").get_logger()
        >>> log.info("Starting process...")
        >>> log.error("An error occurred: %s", error_msg)
        """
        return self.logger
