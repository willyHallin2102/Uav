"""
    logs / logger.py
    ----------------
    A simple wrapper of Python's logging module that provides a console logger
    with some increased ability to control the output console messages with 
    preemptive coloring compared to printing the messages.
"""
from __future__ import annotations

import logging
import sys
import threading

from enum import IntEnum



class Level(IntEnum):
    """
    Logging level matching the python's logging module, it does provide 
    a accessible object to alter the messages that is to be console 
    logged based on the severity assigned.
    """
    DEBUG = logging.DEBUG           # 10
    INFO = logging.INFO             # 20
    WARNING = logging.WARNING       # 30
    ERROR = logging.ERROR           # 40
    CRITICAL = logging.CRITICAL     # 50


class Colors:
    """
    The coloring for preemptive purposes within the console whenever 
    logging the message attached to a particular severity of the message
    being logged. This class define these ANSI colors.
    """
    RESET = "\033[0m"
    GREEN = "\033[32m"
    CYAN = "\033[36m"
    YELLOW = "\033[33m"
    RED = "\033[31m"

# Collect all the severity coloring attachments into a dictionary
LEVEL_COLOR = {
    Level.DEBUG: Colors.GREEN,
    Level.INFO: Colors.CYAN,
    Level.WARNING: Colors.YELLOW,
    Level.ERROR: Colors.RED,
    Level.CRITICAL: Colors.RED
}




class ColorFormatter(logging.Formatter):
    """
    Console logger meant to attach the Level Coloring to the console messages.
    """

    def format(self, record):

        # Get the level-color
        level = Level(record.levelno)
        color = LEVEL_COLOR.get(level, Colors.RESET)

        # Add color to the level name
        levelname = record.levelname
        record.levelname = f"{color}{levelname:<8}{Colors.RESET}"

        # Format the message
        message = super().format(record)
        record.levelname = levelname

        return message



class Logger:
    """
    A simple wrapper of logging ``python module`` with colored console output. This
    provides a interface and a reliable logging.
    """
    _instances = {}
    _lock = threading.Lock()
    _configured = False


    def __new__(cls, name: str = "logger", level: Level = Level.INFO):

        with cls._lock:

            if name not in cls._instances:
                instance = super().__new__(cls)
                cls._instances[name] = instance
            
            else:
                instance = cls._instances[name]
                instance._logger.setLevel(level)
        
        return cls._instances[name]
    

    def __init__(self, name: str = "logger", level: Level = Level.INFO):
        """
            initialize the Logger instance
        """

        # Setup only if not instantiated
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self.name = name

        # Configure root logger first time only
        if not Logger._configured:
            self._configure_root_logger()
        
        # Create/get the underlying logger 
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)

        # Add console formatter handler
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(level)

            # Console coloring formatter 
            handler.setFormatter(ColorFormatter(
                '%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(funcName)s() | %(message)s',
                datefmt='%H:%M:%S.%f'[:-3]
            ))
            self._logger.addHandler(handler)
            self._logger.propagate = False
    

    @classmethod
    def _configure_root_logger(cls):
        """
        Configuring the root should only be performed once, this should therefore 
        be called first time logger is constructed, ``_configured = False``, else 
        skip this configuration step.
        """

        # Remove all potential existing handlers ``duplicate messages``
        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)
        
        # Set root-level to Level.DEBUG to allow all messages
        root.setLevel(logging.DEBUG)
        cls._configured = True
    

    @property
    def level(self) -> Level:
        """
        Get the current logging level
        """
        return Level(self._logger.level)
    
    @level.setter
    def level(self, level: Level):
        """
        Set the logging level dynamically
        """
        self._logger.setLevel(level)

        # Update handlers
        for handler in self._logger.handlers:
            handler.setLevel(level)
    

    def set_level(self, level: Level):
        """
        Set the logging level dynamically.
        
        -----
        Args:
        level: New minimum severity level
        """
        self.level = level
    

    def debug(self, msg):
        self._logger.debug(msg)
    
    def info(self, msg):
        self._logger.info(msg)
    
    def warning(self, msg):
        self._logger.warning(msg)
    
    def error(self, msg):
        self._logger.error(msg)
    
    def critical(self, msg):
        self._logger.critical(msg)
    

    def disable(self):
        """
        Disable all messaging
        """
        self.set_level(Level.CRITICAL + 1)
    
    def enable(self, level: Level = Level):
        self.set_level(level)
    

    def get_logger(self):
        """
        Get the underlying logging.logger instance
        """
        return self._logger
