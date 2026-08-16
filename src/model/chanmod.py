"""
    src / model / chanmod.py
    ------------------------
    Channel model acting as the main operator, manages the link state predictor
    model along with the path model. All operation of path modeling is abstracted
    to the path model solely, only retain results to operate the intermediate
    between the link predictions made to pass this to the path model which takes 
    over and resort to black--box the modeling.
"""
from __future__ import annotations

import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = 2

import numpy as np
import tensorflow as tf

from pathlib import Path
from typing import Dict, List, Tuple, Union

from src.cfg.data import DataConfig
from src.model.link import LinkStatePredictor



class ChannelModel:

    def __init__(self,
        directory: Union[str, Path], config: DataConfig = DataConfig(),
        model_type: str = "vae", seed: int = 42
    ):
        """
            Initialize Channel Model Instance
        """
        self.dir = Path(__file__).resolve().parents[2] / "models" / directory
        self.dir.mkdir(parents=True, exist_ok=True)

        self.link = LinkStatePredictor(
            directory=Path(self.dir) / "link", rx_types=config.rx_types,
            n_unit_links=config.n_unit_links, dropout_rate=config.dropout_rate
        )
