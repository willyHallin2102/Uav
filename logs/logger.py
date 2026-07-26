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
    RESET = "\033[0m"
    GREEN = "\033[32m"
    CYAN = "\033[36m"
    YELLOW = "\033[33m"
    RED = "\033[31m"



class ANSI:
    RESET = "\033[0m"

    class FG:
        BLACK   = "\033[30m"
        RED     = "\033[31m"
        GREEN   = "\033[32m"
        YELLOW  = "\033[33m"
        BLUE    = "\033[34m"
        MAGENTA = "\033[35m"
        CYAN    = "\033[36m"
        WHITE   = "\033[37m"

    class BG:
        BLACK   = "\033[40m"
        RED     = "\033[41m"
        GREEN   = "\033[42m"
        YELLOW  = "\033[43m"
        BLUE    = "\033[44m"
        MAGENTA = "\033[45m"
        CYAN    = "\033[46m"
        WHITE   = "\033[47m"

    class Style:
        BOLD       = "\033[1m"
        DIM        = "\033[2m"
        ITALIC     = "\033[3m"
        UNDERLINE  = "\033[4m"
        BLINK      = "\033[5m"
        REVERSE    = "\033[7m"
        STRIKE     = "\033[9m"


LEVEL_STYLE = {
    Level.DEBUG:
        ANSI.FG.GREEN,

    Level.INFO:
        ANSI.FG.CYAN,

    Level.WARNING:
        ANSI.Style.BOLD +
        ANSI.FG.YELLOW,

    Level.ERROR:
        ANSI.Style.BOLD +
        ANSI.FG.RED,

    Level.CRITICAL:
        ANSI.Style.BOLD +
        ANSI.Style.UNDERLINE +
        ANSI.BG.RED +
        ANSI.FG.WHITE,
}


LEVEL_ICON = {
    Level.DEBUG     : "🔧",
    Level.INFO      : "ℹ️  ",    # Normalize to align console
    Level.WARNING   : "⚠️  ",    # -''- -''- -''- -''- -''-
    Level.ERROR     : "❌",
    Level.CRITICAL  : "💥",
}



class ColorFormatter(logging.Formatter):
    """
    Console logger meant to attach the Level Coloring to the console messages.
    """

    def format(self, record):
        level = Level(record.levelno)

        original_levelname = record.levelname
        original_icon = getattr(record, "icon", None)
        record.icon = f"{LEVEL_ICON.get(level, '❓'):<2}"

        record.levelname = (
            f"{LEVEL_STYLE.get(level, ANSI.RESET)}"
            f"{original_levelname:<8}"
            f"{ANSI.RESET}"
        )

        try:
            return super().format(record)

        finally:
            # Restore original values
            record.levelname = original_levelname
            record.icon = original_icon



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

        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(level)

            # Create formatter with icon field
            # formatter = ColorFormatter(
            #     '%(asctime)s | %(icon)s%(levelname)s | %(filename)s:%(lineno)d | %(funcName)s() | %(message)s'
            # )
            formatter = ColorFormatter(
                # "%(asctime)s | "
                "%(icon)s %(levelname)-10s | "
                "%(filename)s:%(lineno)-4d | "
                "%(funcName)-16s | "
                "%(message)s"
            )
            handler.setFormatter(formatter)
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
    
    # Stacklevel solves the problem of console referring these, now 
    # it does refer to the intended function.
    def debug(self, msg):
        self._logger.debug(msg, stacklevel=2)
    
    def info(self, msg):
        self._logger.info(msg, stacklevel=2)
    
    def warning(self, msg):
        self._logger.warning(msg, stacklevel=2)
    
    def error(self, msg):
        self._logger.error(msg, stacklevel=2)
    
    def critical(self, msg):
        self._logger.critical(msg, stacklevel=2)
    

    def disable(self):
        """
        Disable all messaging
        """
        self.set_level(Level.CRITICAL + 1)
    
    def enable(self, level: Level = Level.INFO):
        self.set_level(level)
    

    def get_logger(self):
        """
        Get the underlying logging.logger instance
        """
        return self._logger

