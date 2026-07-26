"""
    tests / logger.py
    -----------------
    Test script for the logger class, including initialization and 
    the various severity tests, color formatting, caller detection, and messaging.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import io
import logging
import sys
import threading
import time
import re

from logs.logger import Logger, Level, Colors
from typing import Any, List

from tools.utilities import runner, builder, CommandSpec



# ======================================================================
#       Testing Helpers
# ======================================================================

def get_logger(name: str = "Test-Logger", level: Level = Level.INFO):
    """Create a fresh logger instance for testing"""
    with Logger._lock:

        if name in Logger._instances:
            del Logger._instances[name]
    
    return Logger(name, level)



def capture_output(f, *args, **kwargs):
    """
    Capture stdout output from a certain function call
    """
    outout = io.StringIO()
    stdout = sys.stdout

    handlers = logging.root.handlers[:]
    try:
        sys.stdout = output

        for handler in logging.root.handlers:
            logging.root.removeHandler(handler)
        
        f(*args, **kwargs)
        return outout.getvalue()
    
    finally:
        sys.stdout = stdout

        # Restore all the handlers
        for handler in handlers:
            logging.root.addHandler(handler)



# ======================================================================
#       Testing Methods
# ======================================================================

def test_initialization(args: argparse.Namespace):
    """Test logger initialization"""
 
    with Logger._lock:
        Logger._instances.clear()
        Logger._configured = False
    
    logger = Logger("Tester")
    print(f"Logger Instance: {logger}")
    print(f"Logger name: {logger.name}")
    print(f"Default level: {logger.level}")
    
    logger_debug = Logger("DebugLogger", Level.DEBUG)
    print(f"Debug logger level: {logger_debug.level}")
    
    print("\n✅ Initialization test passed")



def test_singleton(args: argparse.Namespace):
    """Test singleton behavior"""
    with Logger._lock:
        Logger._instances.clear()
        Logger._configured = False
    
    logger1, logger2 = Logger("TestSingleton"), Logger("TestSingleton")
    
    print(f"\tlogger1 id: {id(logger1)}")
    print(f"\tlogger2 id: {id(logger2)}")
    print(f"\tSame object: {logger1 is logger2}")
    
    print("\n2. ---------- Testing different named instances ----------")
    logger_a, logger_b = Logger("LoggerA"), Logger("LoggerB")
    
    print(f"\tlogger_a id: {id(logger_a)}")
    print(f"\tlogger_b id: {id(logger_b)}")
    print(f"\tDifferent objects: {logger_a is not logger_b}")
    
    print("\n3. ---------- Testing instance cache ----------")
    with Logger._lock:
        Logger._instances.clear()
        Logger._configured = False
    
    logger_new = Logger("TestCache")
    print(f"\tCache size: {len(Logger._instances)}")
    print(f"\tCached names: {list(Logger._instances.keys())}")
    
    print("\n✅ Singleton test passed")



def test_logging_concurrent(args: argparse.Namespace):
    """Test concurrent logging from multiple threads"""
    print("1. ---------- Concurrent logging multiple threads ----------")    
    logger = get_logger("Concurrent-Logger")
    
    def log_messages(thread_id: int, count: int = 100):
        
        for i in range(count):
            level = [
                Level.DEBUG, Level.INFO, Level.WARNING, 
                Level.ERROR, Level.CRITICAL
            ][i % 100]
            message = f"Thread {thread_id} message {i}"
            
            if level == Level.DEBUG:
                logger.debug(message)
            
            elif level == Level.INFO:
                logger.info(message)
            
            elif level == Level.WARNING:
                logger.warning(message)
            
            elif level == Level.ERROR:
                logger.error(message)
            
            else:
                logger.critical(message)
            
            time.sleep(0.001)
    
    threads = []
    for i in range(100):
        t = threading.Thread(target=log_messages, args=(i, 100))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    print("\tThreads created: 5, messages each: 10, total: 50")
    print("\n2. ---------- Logger thread safety with instance sharing ----------")
    
    shared_logger = Logger("SharedThreadLogger")
    def shared_log(thread_id: int):
        for i in range(100):
            shared_logger.info(f"Shared log from thread {thread_id}, msg {i}")
            time.sleep(0.001)
    
    threads2 = []
    for i in range(100):
        t = threading.Thread(target=shared_log, args=(i,))
        threads2.append(t)
        t.start()
    
    for t in threads2:
        t.join()
    
    print("\n✅ Concurrent logging test passed")



def test_logging_levels(args: argparse.Namespace):
    """
    Test all logging levels and their output
    """
    print("1. ---------- Testing all Logging Outputs ----------")
    
    print("\nTesting with default INFO level (should show INFO and above):")
    logger = get_logger("LevelTest", Level.INFO)
    
    logger.debug("DEBUG message (should NOT show)")
    logger.info("INFO message (should show)")
    logger.warning("WARNING message (should show)")
    logger.error("ERROR message (should show)")
    logger.critical("CRITICAL message (should show)")
    
    print("\nTesting with DEBUG level (should show all):")
    logger = get_logger("DebugTest", Level.DEBUG)
    
    logger.debug("DEBUG message (should show)")
    logger.info("INFO message (should show)")
    logger.warning("WARNING message (should show)")
    logger.error("ERROR message (should show)")
    logger.critical("CRITICAL message (should show)")
    
    print("\nTesting with ERROR level (should show ERROR and above):")
    logger = get_logger("ErrorTest", Level.ERROR)
    
    logger.debug("DEBUG message (should NOT show)")
    logger.info("INFO message (should NOT show)")
    logger.warning("WARNING message (should NOT show)")
    logger.error("ERROR message (should show)")
    logger.critical("CRITICAL message (should show)")
    
    print("\n✅ Logging levels test passed")



def test_message_types(args: argparse.Namespace):
    """Test logging different message types"""
    print("1. ---------- Testing different message types ----------")
    
    logger = get_logger("TypeTest", Level.DEBUG)
    
    print("\nLogging various message types:")
    logger.info("String message")
    logger.info(42)
    logger.info([1, 2, 3])
    logger.info({"key": "value"})
    logger.info(None)
    logger.info(3.14159)
    logger.info(True)
    
    print("\n✅ Message types test passed")


def test_level_control(args: argparse.Namespace):
    """Test dynamic level control"""
    print("1. ---------- Testing dynamic level control ----------")
    
    logger = get_logger("ControlTest", Level.INFO)
    
    print("\nInitial level: INFO")
    logger.debug("DEBUG - should NOT show")
    logger.info("INFO - should show")
    
    print("\nChanging to DEBUG level")
    logger.set_level(Level.DEBUG)
    logger.debug("DEBUG - should show now")
    logger.info("INFO - should show")
    
    print("\nChanging to ERROR level")
    logger.set_level(Level.ERROR)
    logger.debug("DEBUG - should NOT show")
    logger.info("INFO - should NOT show")
    logger.warning("WARNING - should NOT show")
    logger.error("ERROR - should show")
    
    print("\nTesting disable/enable")
    logger.disable()
    print("Disabled - nothing should show below:")
    logger.info("This should NOT show")
    
    logger.enable(Level.WARNING)
    print("Enabled with WARNING level:")
    logger.info("This should NOT show")
    logger.warning("This should show")
    
    print("\n✅ Level control test passed")


def test_colors(args: argparse.Namespace):
    """Test color formatting"""
    print("1. ---------- Testing color formatting ----------")
    
    logger = get_logger("ColorTest", Level.DEBUG)
    
    print("\nLogging with colors (ANSI codes should be visible):")
    logger.debug("DEBUG - Cyan")
    logger.info("INFO - Green")
    logger.warning("WARNING - Yellow")
    logger.error("ERROR - Red")
    logger.critical("CRITICAL - Red")
    
    print("\n✅ Color test passed")


def test_get_logger(args: argparse.Namespace):
    """Test getting the underlying logger"""
    print("1. ---------- Testing get_logger() ----------")
    
    logger = get_logger("WrapperTest")
    underlying = logger.get_logger()
    
    print(f"Logger wrapper: {type(logger).__name__}")
    print(f"Underlying logger: {type(underlying).__name__}")
    print(f"Is instance of logging.Logger: {isinstance(underlying, logging.Logger)}")
    print(f"Logger name: {underlying.name}")
    
    underlying.info("Message from underlying logger")
    print("\n✅ get_logger test passed")


def test_caller_detection(args: argparse.Namespace):
    """Test caller detection (now handled by logging module)"""
    print("\n1. ---------- Testing caller detection ----------")
    
    logger = get_logger("CallerTest", Level.DEBUG)
    
    # Calling from within a class instance
    class TestClass:
        def __init__(self):
            self.logger = logger
        
        def test_method(self):
            self.logger.info("Message from class method")
    
    c = TestClass()
    c.test_method()
    
    print("\n2. ---------- Testing function caller detection ----------")
    
    def test_function():
        logger.info("Message from function")
    
    test_function()
    
    print("\n3. ---------- Testing nested function detection ----------")
    
    def outer_function():
        def inner_function():
            logger.info("Message from nested function")
        inner_function()
    
    outer_function()
    
    print("\n✅ Caller Detection Passed")


# ======================================================================
#       Main Runner
# ======================================================================

COMMON = [
    {"flags": ["--n-samples", "-n"], "kwargs": {"type": int, "default": 100}},
    {"flags": ["--n-perf-samples"], "kwargs": {"type": int, "default": 1000}},
    {"flags": ["--verbose", "-v"], "kwargs": {"action": "store_true"}},
]
TEST_ARGS = [*COMMON]


@runner
def main():
    p = builder([
        CommandSpec(
            "init", "Testing initialize a logger instance",
            test_initialization, TEST_ARGS,
        ),
        CommandSpec(
            "singleton", "Testing the singleton pattern",
            test_singleton, TEST_ARGS,
        ),
        CommandSpec(
            "concurrent", "Test concurrent logging",
            test_logging_concurrent, TEST_ARGS
        ),
        CommandSpec(
            "levels", "Test logging severity levels",
            test_logging_levels, TEST_ARGS,
        ),
        CommandSpec(
            "message_types", "Logging different types in messages",
            test_message_types, TEST_ARGS,
        ),
        CommandSpec(
            "level_control", "Test dynamic level control",
            test_level_control, TEST_ARGS,
        ),
        CommandSpec(
            "colors", "Test color formatting",
            test_colors, TEST_ARGS,
        ),
        CommandSpec(
            "get_logger", "Test getting underlying logger",
            test_get_logger, TEST_ARGS,
        ),
        CommandSpec(
            "caller", "Testing retrieval of caller id of function calls",
            test_caller_detection, TEST_ARGS
        ),
    ])

    args = p.parse_args()
    args._handler(args)



if __name__ == "__main__":
    main()