"""
    src / cfg / data.py
    -------------------
    Manages the data -- parameters structures of teh environment rather than the
    UAV, that includes the surrounding e.g., frequency, decibel power--loss of 
    the carrier signal.
"""
from __future__ import annotations

import datetime as dt
from typing import ClassVar, Final, Tuple
from enum import IntEnum
from dataclasses import dataclass, field

from src.cfg.const import TERRESTRIAL, AERIAL



class LinkState(IntEnum):
    """
    Link configuration setup, defining the states and operations to retain the
    number of plausible number of states, retrieve their corresponding names 
    and other additional actual states simply added.
    """
    NO_LINK = 0
    NLOS    = 1
    LOS     = 2

    @classmethod
    @property
    def n_states(cls) -> int:
        return len(cls)
    

    @classmethod
    def from_string(cls, state: str) -> LinkState:
        return {
            "no-link": cls.NO_LINK, "nlos": cls.NLOS, "los": cls.LOS
        }[state.lower()]



class DataConfig:
    """
    """
    frequency   : float = 28e+9
    created     : str = field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat()
    )
    description : str = "dataset"
    rx_types    : Tuple[str, ...] = (TERRESTRIAL, AERIAL)
    max_path_loss   : float = 200.0
    tx_power_dm : float = 16.0
    n_max_paths : int = 20
    n_unit_links    : Tuple[int, ...] = (25, 10)
    add_zero_los_frac   : float = 0.10
    dropout_rate    : float = 0.20

    def __post_init__(self):
        """
        Check that no invalid parameters are assigned
        """
        if self.frequency <= 0: 
            raise ValueError("Frequency must be positive")
        
        if self.max_path_loss <= 0.0: 
            raise ValueError("Maximum path loss must be positive")
        
        if self.n_max_paths <= 0: 
            raise ValueError("Number of max path must be positive")
        
        if any(n <= 0 for n in self.n_unit_links):
            raise ValueError("All unit links must be positive")
        
        if not 0 <= self.add_zero_los_frac <= 1:
            raise ValueError("LOS fraction must be between 0 and 1")
        
        if not 0 <= self.dropout_rate <= 1:
            raise ValueError("Dropout rate must be between 0 and 1")
    
    @property
    def wavelength(self) -> float: 
        return LIGHT_SPEED / self.frequency
