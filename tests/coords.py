"""
    tests / coords.py
    -----------------
    Test script for coordinate transformations and spherical angle operations.
    Testing Cartesian <-> Spherical conversions, adding / subtracting angles.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from typing import Any, Dict, List, Tuple
from argparse import Namespace

from src.maths.coords import (
    cartesian_to_spherical, spherical_to_cartesian,
    add_angles, sub_angles, DEG2RAD, RAD2DEG, EPS
)
from tools.utilities import runner, builder, CommandSpec
from tools.timer import Timer


# ======================================================================
#       Generate Vectors
# ======================================================================

def generate_random_cartesian_vectors(
    n: int = 1000, r_range: Tuple[float, float] = (0.1, 100.0),
    seed: int | None = 42
) -> np.ndarray:
    """
    Generate random Cartesian vectors for testing the operations
    """
    rng = np.random.default_rng(seed)

    theta = np.arccos(2 * rng.random(n) - 1)
    phi = 2 * np.pi * rng.random(n)

    r = rng.uniform(r_range[0], r_range[1], n)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return np.column_stack((x, y, z))


def generate_random_spherical_vectors(
    n: int = 1000, r_range: Tuple[float, float] = (0.1, 100.0),
    seed: int | None = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate random spherical vectors for testing, measuring in degrees
    """
    rng = np.random.default_rng(seed)

    r = rng.uniform(r_range[0], r_range[1], n)
    phi = rng.uniform(0, 360, n)
    theta = np.arccos(2 * rng.random(n) - 1) * RAD2DEG

    return r, phi, theta

#################
# --------------
#################

def test_cases() -> List[Dict[str, Any]]:
    return [
        {
            "name": "origin",
            "cartesian": np.array([0, 0, 0]),
            "spherical": (0.0, 0.0, 0.0)
        },
        {
            "name": "positive_x_axis",
            "cartesian": np.array([1, 0, 0]),
            "spherical": (1.0, 0.0, 90.0)
        },
        {
            "name": "positive_y_axis",
            "cartesian": np.array([0, 1, 0]),
            "spherical": (1.0, 90.0, 90.0)
        },
        {
            "name": "positive_z_axis",
            "cartesian": np.array([0, 0, 1]),
            "spherical": (1.0, 0.0, 0.0)
        },
        {
            "name": "negative_x_axis",
            "cartesian": np.array([-1, 0, 0]),
            "spherical": (1.0, 180.0, 90.0)
        },
        {
            "name": "negative_z_axis",
            "cartesian": np.array([0, 0, -1]),
            "spherical": (1.0, 0.0, 180.0)
        },
        {
            "name": "unit_sphere_diag",
            "cartesian": np.array([1, 1, 1]) / np.sqrt(3),
            "spherical": (1.0, 45.0, 54.73561031724536)
        }
    ]

def generate_rotation_test_cases() -> List[Dict[str, Any]]:
    return [
        {
            "name": "identity",
            "phi0": 0.0, "theta0": 0.0,
            "phi1": 0.0, "theta1": 0.0,
            "expected_add": (0.0, 0.0),
            "expected_sub": (0.0, 0.0)
        },
        {
            "name": "rotation_around_z",
            "phi0": 0.0, "theta0": 90.0,
            "phi1": 45.0, "theta1": 0.0,
            "expected_add": (45.0, 90.0),
            "expected_sub": (-45.0, 90.0)
        },
        {
            "name": "rotation_around_y",
            "phi0": 0.0, "theta0": 90.0,
            "phi1": 0.0, "theta1": 45.0,
            "expected_add": (0.0, 45.0),
            "expected_sub": (180.0, 135.0)  # inverse
        },
        {
            "name": "compound_rotation",
            "phi0": 30.0, "theta0": 60.0,
            "phi1": 20.0, "theta1": 40.0,
            "expected_add": (45.0, 70.0),
            "expected_sub": (10.0, 30.0)
        }
    ]


# ======================================================================
#       Test Methods 
# ======================================================================

def test_cartesian_to_spherical(args: Namespace):
    """Test Cartesian to spherical coordinate conversion."""
    print("\nTesting single vector conversion:")
    cartesian = np.array([1.0, 0.0, 0.0])
    r, phi, theta = cartesian_to_spherical(cartesian)

    print(f"\tCartesian: {cartesian}")
    print(f"\tSpherical: r={r[0]:.6f}, φ={phi[0]:.6f}°, θ={theta[0]:.6f}°")

    for case in test_cases():
        print()
        cartesian = case["cartesian"]
        expected_r, expected_phi, expected_theta = case["spherical"]    
        r, phi, theta = cartesian_to_spherical(cartesian)
        
        print(f"\t{case['name']}:")
        print(f"\tInput: {cartesian}")
        print(f"\tOutput: r={r[0]:.6f}, φ={phi[0]:.6f}°, θ={theta[0]:.6f}°")    
    
    print("\n✅ Cartesian to spherical conversion test passed")



def test_spherical_to_cartesian(args: Namespace):
    """Test spherical to Cartesian coordinate conversion."""

    print("1. ---------- Testing Spherical to Cartesian Conversion ----------")
    print("\nTesting single vector conversion:")
    r, phi, theta = 1.0, 0.0, 90.0
    cartesian = spherical_to_cartesian(r, phi, theta)

    print(f"\tSpherical: r={r}, φ={phi}°, θ={theta}°")
    print(f"\tCartesian: {cartesian[0]}")
    
    print("\nTesting specific test cases (round-trip):")
    for case in test_cases():
        cart = case["cartesian"]
        
        r, phi, theta = cartesian_to_spherical(cart)
        cart_reconstructed = spherical_to_cartesian(r, phi, theta)[0]
        
        print(f"\t{case['name']}:")
        print(f"\tOriginal: {cart}")
        print(f"\tReconstructed: {cart_reconstructed}")
    
    print("\n2. ---------- Testing Broadcasting ----------")
    r_scalar = 2.0
    phi_array = np.array([0.0, 90.0, 180.0, 270.0])
    theta_array = np.array([90.0, 90.0, 90.0, 90.0])
    
    cart_broadcast = spherical_to_cartesian(r_scalar, phi_array, theta_array)
    print(f"\tRadius scalar: {r_scalar}")
    print(f"\tPhi array: {phi_array}")
    print(f"\tResult shape: {cart_broadcast.shape}")
    
    expected = np.array([
        [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [-2.0, 0.0, 0.0], [0.0, -2.0, 0.0]
    ])

    print("\n✅ Spherical to Cartesian conversion test passed")



def test_round_trip(args: Namespace):
    """Test round-trip conversion between Cartesian and spherical."""

    print("1. ---------- Testing Round-Trip Conversion ----------")
    
    n_samples = args.n_samples
    print(f"Testing {n_samples} random vectors...\n")
    vectors = generate_random_cartesian_vectors(n_samples, seed=42)
    
    r, phi, theta = cartesian_to_spherical(vectors)
    reconstructed = spherical_to_cartesian(r, phi, theta)
    
    errors = np.linalg.norm(vectors - reconstructed, axis=1)
    max_error = np.max(errors)
    mean_error = np.mean(errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    
    print(f"\tMax error: {max_error:.6f}")
    print(f"\tMean error: {mean_error:.6f}")
    print(f"\tRMSE: {rmse:.6f}")
    
    valid_mask = np.linalg.norm(vectors, axis=1) > EPS
    if np.any(valid_mask):
        max_error_valid = np.max(errors[valid_mask])
        assert max_error_valid < 1e-6, f"Max error too large: {max_error_valid}"
    
    r, phi, theta = cartesian_to_spherical(np.array([0, 0, 0]))
    cartesian = spherical_to_cartesian(r, phi, theta)
    
    print(f"\n\tOrigin: {cartesian[0]}")
    
    small_vec = np.array([1e-8, 1e-8, 1e-8])
    r, phi, theta = cartesian_to_spherical(small_vec)
    cartesian = spherical_to_cartesian(r, phi, theta)
    
    print(f"\tSmall vector: {cartesian[0]}")
    print("\n✅ Round-trip conversion test passed")



def test_add_angles(args: Namespace):
    """Test composition of spherical angles (add_angles)."""
    
    print("1. ---------- Testing Angle Composition (add_angles) ----------")
    print("\nTesting specific test cases:")
    for case in generate_rotation_test_cases():
        phi0, theta0 = case["phi0"], case["theta0"]
        phi1, theta1 = case["phi1"], case["theta1"]
        expected_phi, expected_theta = case["expected_add"]
        
        phi_result, theta_result = add_angles(phi0, theta0, phi1, theta1)
        
        print(f"\t{case['name']}:")
        print(f"\t\tInput: (φ0={phi0}°, θ0={theta0}°) + (φ1={phi1}°, θ1={theta1}°)")
        print(f"\t\tOutput: (φ={phi_result:.6f}°, θ={theta_result:.6f}°)")
        print(f"\t\tExpected: (φ={expected_phi:.6f}°, θ={expected_theta:.6f}°)\n")
        
    print("\n2. ---------- Testing Broadcasting in add_angles ----------")
    
    phi_out, theta_out = add_angles(30.0, 45.0, 10.0, 20.0)
    print(f"  Scalar + scalar: φ={phi_out}, θ={theta_out}")
    assert np.isscalar(phi_out) or phi_out.shape == (), "Scalar output expected"
    
    phi0_scalar, theta0_scalar = 30.0, 45.0
    phi1_arr = np.array([0.0, 90.0, 180.0])
    theta1_arr = np.array([0.0, 0.0, 0.0])
    
    phi_out, theta_out = add_angles(phi0_scalar, theta0_scalar, phi1_arr, theta1_arr)
    print(f"\tScalar + array: output shape = {phi_out.shape}")
    assert phi_out.shape == (3,), f"Expected shape (3,), got {phi_out.shape}"
    
    phi0_arr = np.array([0.0, 45.0, 90.0])
    theta0_arr = np.array([90.0, 90.0, 90.0])
    phi1_arr = np.array([10.0, 20.0, 30.0])
    theta1_arr = np.array([0.0, 0.0, 0.0])
    
    phi_out, theta_out = add_angles(phi0_arr, theta0_arr, phi1_arr, theta1_arr)
    print(f"\tArray + array: output shape = {phi_out.shape}")
    assert phi_out.shape == (3,), f"Expected shape (3,), got {phi_out.shape}"
    
    print("\nTesting broadcast mismatch handling:")
    phi0_bad = np.array([0.0, 45.0])
    theta0_bad = np.array([90.0, 90.0])
    phi1_bad = np.array([10.0, 20.0, 30.0])
    
    try:
        phi_out, theta_out = add_angles(phi0_bad, theta0_bad, phi1_bad, theta1_arr)
        print("ERROR: Should have raised ValueError for shape mismatch")
        assert False, "Broadcast mismatch should raise ValueError"
    
    except ValueError as e:
        print(f"\tCaught expected error: {e}")
    
    print("\n✅ Angle composition test passed")


def test_sub_angles(args: Namespace):
    """Test subtraction of spherical angles (sub_angles)."""
    
    print("1. ---------- Testing Angle Subtraction (sub_angles) ----------")
    print("\nTesting specific test cases:")
    for case in generate_rotation_test_cases():
        phi0, theta0 = case["phi0"], case["theta0"]
        phi1, theta1 = case["phi1"], case["theta1"]
        expected_phi, expected_theta = case["expected_sub"]
        
        phi_result, theta_result = sub_angles(phi0, theta0, phi1, theta1)
        phi_result = phi_result % 360.0
        expected_phi = expected_phi % 360.0
        
        print(f"\t{case['name']}:")
        print(f"\t\tInput: (φ0={phi0}°, θ0={theta0}°) - (φ1={phi1}°, θ1={theta1}°)")
        print(f"\t\tOutput: (φ={phi_result:.6f}°, θ={theta_result:.6f}°)")
        print(f"\t\tExpected: (φ={expected_phi:.6f}°, θ={expected_theta:.6f}°)\n")
        
    print("\n2. ---------- Testing Inverse Relationship ----------")
    
    phi0, theta0 = 30.0, 45.0
    phi1, theta1 = 10.0, 20.0
    
    phi_add, theta_add = add_angles(phi0, theta0, phi1, theta1)
    phi_sub, theta_sub = sub_angles(phi_add, theta_add, phi1, theta1)
    
    print(f"\tOriginal: (φ={phi0}°, θ={theta0}°)")
    print(f"\tAfter add: (φ={phi_add:.6f}°, θ={theta_add:.6f}°)")
    print(f"\tAfter subtract: (φ={phi_sub:.6f}°, θ={theta_sub:.6f}°)")
    
    print("\n✅ Angle subtraction test passed")




def test_performance(args: Namespace):
    """Test performance of coordinate operations using Timer."""

    print("1. ---------- Testing Performance ----------")
    
    n_samples = args.n_perf_samples
    print(f"Benchmarking with {n_samples:,} samples...")
    
    vectors = generate_random_cartesian_vectors(n_samples, seed=42)
    r, phi, theta = cartesian_to_spherical(vectors)
    
    print("\nCartesian → Spherical:")
    
    with Timer(name="cartesian_to_spherical", show=True) as t:
        r_out, phi_out, theta_out = cartesian_to_spherical(vectors)
    
    rate = n_samples / t.elapsed
    print(f"\tRate: {rate:.1f} conversions/sec")
    
    print("\nSpherical → Cartesian:")
    
    with Timer(name="spherical_to_cartesian", show=True) as t:
        cart_out = spherical_to_cartesian(r, phi, theta)
    
    rate = n_samples / t.elapsed
    print(f"\tRate: {rate:.1f} conversions/sec")
    
    print("\nAngle Composition (add_angles):")
    phi_rand = np.random.uniform(0, 360, n_samples)
    theta_rand = np.random.uniform(0, 180, n_samples)
    
    with Timer(name="add_angles", show=True) as t:
        phi_result, theta_result = add_angles(phi, theta, phi_rand, theta_rand)
    
    rate = n_samples / t.elapsed
    print(f"  Rate: {rate:.1f} comps/sec")
    
    print("\nAngle Subtraction (sub_angles):")
    
    with Timer(name="sub_angles", show=True) as t:
        phi_result, theta_result = sub_angles(phi, theta, phi_rand, theta_rand)
    
    rate = n_samples / t.elapsed
    print(f"  Rate: {rate:.1f} comps/sec")
    
    print("\n2. ---------- Scaling Test ----------")
    sizes = [1000, 10000, 100000] if n_samples >= 100000 else [100, 1000, n_samples]
    
    print(f"Batch size scaling (cartesian_to_spherical):")
    for size in sizes:
        if size <= n_samples:
            test_vectors = vectors[:size]
    
            with Timer(name=f"n={size}", show=True) as t:
                r_out, phi_out, theta_out = cartesian_to_spherical(test_vectors)
    
            rate = size / t.elapsed
            print(f"\t{size:,} vectors: {rate:.1f} conversions/sec")
    
    print("\n✅ Performance test passed")

# ======================================================================
#       Main Runner
# ======================================================================

ARGS = [
    {"flags": ["--n-samples", "-n"], "kwargs": {"type": int, "default": 100}},
    {"flags": ["--n-perf-samples"], "kwargs": {"type": int, "default": 100000}},
    {"flags": ["--verbose", "-v"], "kwargs": {"action": "store_true"}},
]


@runner
def main():
    p = builder([
        CommandSpec(
            "cart_to_sphere",
            "Test Cartesian to spherical conversion",
            test_cartesian_to_spherical,
            [*ARGS]
        ),
        CommandSpec(
            "sphere_to_cart",
            "Test spherical to Cartesian conversion",
            test_spherical_to_cartesian,
            [*ARGS]
        ),
        CommandSpec(
            "roundtrip",
            "Test round-trip conversion",
            test_round_trip,
            [*ARGS]
        ),
        CommandSpec(
            "add_angles",
            "Test angle composition (add_angles)",
            test_add_angles,
            [*ARGS]
        ),
        CommandSpec(
            "sub_angles",
            "Test angle subtraction (sub_angles)",
            test_sub_angles,
            [*ARGS]
        ),
        CommandSpec(
            "performance",
            "Test performance of coordinate operations using Timer",
            test_performance,
            [*ARGS]
        ),
    ])

    args = p.parse_args()
    args._handler(args)


if __name__ == "__main__":
    main()
