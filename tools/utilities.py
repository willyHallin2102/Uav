"""
    tools / utilities.py
    --------------------
    Simple methods for abstract some functionality from the testing script located 
    in `tests/*.py`. Include, @runner wrapper to enable user key-interruption in 
    case of runtime error, CommandSpec as a container of all cmd commands and 
    information.
"""
import argparse
import logging
import sys
import traceback

from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Type


@dataclass
class CommandSpec:
    """
    Declarative specification of a CLI command
    """
    name: str
    help: str
    handler: Callable
    args: List[Dict[str, Any]] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)

    def __post_init__(self):
        """
            Post Check after Initialization of Requirements
        """
        if not self.name.isidentifier():
            raise ValueError(f"Command name ``{self.name}`` must be valid")
        
        if not callable(self.handler):
            raise TypeError(f"``{self.handler}`` must be callable")
    


def builder(commands: List[CommandSpec]) -> argparse.Namespace:
    """
    Builds the CLI parser from the command specifications provided 
    as the argument
    """
    parser = argparse.ArgumentParser(description="Logger, debug CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    for command in commands:
        p = sub.add_parser(command.name, help=command.help, aliases=command.aliases)

        for arg in command.args:
            p.add_argument(*arg["flags"], **arg["kwargs"])
        
        p.set_defaults(_handler=command.handler)
    
    return parser


def runner(f: Callable) -> Callable:
    """
    Wraps CLI entrypoint with standardized exception handling, avoiding a error
    in runtime, specifically if the getting caught in infinite loop, enable 
    easy exit access.
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        
        # https://emojicombos.com/warning
        except KeyboardInterrupt:
            print("\n⚠️ Aborted by User")
            sys.exit(130)   # Standard exit for SIGINT
        
        # https://emojicombos.com/error
        except Exception as e:
            # Logging the traceback for debugging reasons, what went wrong
            logging.error(f"Test Failed:\``{e}``", exc_info=True)

            print(f"\n⛔ Test Failed: ``{e}``")
            sys.exit(1)
    
    return wrapper



def debug(level: str | None = None):
    """
    Decorator to enable debugging logging for a function
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def wrapper(*args, **kwargs):
            
            logging.debug(f"Callable ``{f.__name__}`` with args={args}, kwargs={kwargs}")
            try:
                result = f(*args, **kwargs)
                logging.debug(f"{f.__name__} returned: {result}")

                return result
            
            except Exception as e:
                logging.error(f"{f.__name__} raised: ``{e}``", exc_info=True)
                raise
        
        return wrapper
    return decorator
