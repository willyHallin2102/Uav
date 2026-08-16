"""
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

import json
import tempfile
import time

from argparse import Namespace
from typing import Dict, Tuple

from src.model.link import LinkStatePredictor
from src.cfg.data import LinkState
from src.cfg.const import AERIAL, TERRESTRIAL

from tools.utilities import runner, builder, CommandSpec


# ======================================================================
#       Helping Functions
# ======================================================================

# Unnecessary -- Load data instead by randomly select entires
def create_link_data(
    n_samples: int = 1_000,
    seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    """
    rx_types = [AERIAL, TERRESTRIAL]
    rng = np.random.default_rng(seed)

    # Generate random position in 3d space, within range [-50,50]
    positions = rng.random((n_samples, 3)) * 100 - 50
    dvec = positions.astype(np.float32)

    rxi = rng.integers(0, len(rx_types), size=n_samples)
    rx_type = np.array([rx_types[i] for i in rxi], dtype=object)

    distances = np.linalg.norm(dvec, axis=1)
    link_state = np.zeros(n_samples, dtype=np.int32)

    for i, distance in enumerate(distances):
        if distance < 20:
            link_state[i] = LinkState.LOS if rng.random() < 0.7 else LinkState.NLOS
        
        elif distance < 50:
            link_state[i] = LinkState.NLOS if rng.random() < 0.7 else LinkState.LOS
        
        else:
            link_state[i] = LinkState.NO_LINK if rng.random() < 0.6 else LinkState.NLOS
    
    return { "dvec": dvec, "rx_type": rx_type, "link_state": link_state }



def get_link_predictor(
    n_unit_links: Tuple[int, ...] = (64, 32),
    dropout_rate: float = 0.2,
) -> LinkStatePredictor:
    """
    Create a fresh LinkStatePredictor instance for testing.
    """
    return LinkStatePredictor(
        rx_types=[AERIAL, TERRESTRIAL],
        n_unit_links=n_unit_links,
        add_zero_frac_los=0.1,
        dropout_rate=dropout_rate,
        directory=Path(tempfile.mkdtemp()) / "link_test",
        seed=42
    )


# ======================================================================
#       Test Method
# ======================================================================

def test_initialization(args: Namespace):
    """
    """
    print("\n1. ---------- Testing Link-State-Predictor ----------")

    predictor = get_link_predictor()
    print(f"LinkStatePredictor instance: {predictor}")
    print(f"Model directory: {predictor.dir}")
    print(f"Receiver types: {predictor.rx_types}")
    print(f"Hidden layer units: {predictor.n_unit_links}")
    print(f"Dropout rate: {predictor.dropout_rate}")
    print(f"Zero LoS fraction: {predictor.add_zero_los_frac}")

    print("\n2. ---------- Build the Link-State Predictor ----------")
    predictor.build()

    print(f"Model Built: {predictor.model is not None}")

    if predictor.model is not None:
        print(f"Model input shape: {predictor.model.input_shape}")
        print(f"Model Output shape: {predictor.model.output_shape}")
        print("\nModel Summary")
        predictor.model.summary()



def test_preprocessing(args: Namespace):
    """
    """
    print("\n1. ---------- Testing Data Preparations ----------")

    predictor = get_link_predictor()
    data = create_link_data(1000)

    # Testing _prepare_arrays 
    xtr, ytr = predictor._prepare_arrays(data, fit = True)
    print(f"Prepared features shape: {xtr.shape}")
    print(f"Label types: {ytr.dtype}")
    print(f"Feature range: [{xtr.min():.3f}, {xtr.max():.3f}]")


    print("\n2. ---------- Testing Transform ----------")
    t1 = predictor._transform_links(
        data["dvec"], data["rx_type"], fit=False
    )
    print(f"Transformed features shape: {t1.shape}")

    t2 = predictor._transform_links(
        data["dvec"], data["rx_type"], fit=False
    )
    assert np.allclose(t1, t2), "Transformation should be deterministic"

    print("\n3. ---------- Testing Zero LoS Augmentation ----------")
    n_samples = len(data["dvec"])
    dvec_aug, rx_aug, link_aug = predictor._add_los_zero(
        data["dvec"], data["rx_type"], data["link_state"]
    )
    print(f"Original number of samples: {n_samples}")
    print(f"Number of augmented samples added: {len(dvec_aug)-n_samples}")

    if len(dvec_aug) > n_samples:
        zero_idx = range(n_samples, len(dvec_aug))
        for i in zero_idx:
            assert np.all(dvec_aug[i][:2] == 0), "dx, dy should be zero"




# ======================================================================
#       Main Runner
# ======================================================================

ARGS = [
    {"flags": ["--n-samples", "-n"], "kwargs": {"type": int, "default": 100}},
]

@runner
def main():
    p = builder([
        CommandSpec(
            "init", "Test LinkStatePredictor and Build",
            test_initialization, [*ARGS]
        ),
        CommandSpec(
            "preproc", "Testing to Prepare arrays",
            test_preprocessing, [*ARGS]
        )
    ])

    args = p.parse_args()
    args._handler(args)


if __name__ == "__main__":
    main()
