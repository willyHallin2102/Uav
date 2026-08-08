"""
    src / maths / coords.py
    -----------------------
    Coordinate transformations and spherical angle operations relates the to the 
    more accurate computations to the AOD and AOA operations. This modules provides
    utilities for specific use--cases.

        - Cartesian <-> spherical coordinate conversions
        - Composition and subtraction of spherical angles, 3d rotations.
        - Batch oriented numerical kernels
    
    All heavy computations are ``numba`` accelerated for efficient executions on
    larger arrays and angles are expressed in degrees while internally operates in 
    radians.

    Conversions:
    ------------
    - Spherical coordinates follow (r, φ, θ):
        r     : radius
        φ     : azimuth angle in degrees (xy-plane, atan2(y, x))
        θ     : inclination angle in degrees (angle from +z axis)
    - Outputs are float64 for numerical stability.
    - Broadcasting is supported where explicitly mentioned.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from numba import njit, prange, float64
from typing import Final, Tuple, Union, overload


AF = npt.NDArray[np.floating]
AF64 = npt.NDArray[np.float64]

DEG2RAD: Final[np.float64] = np.float64(np.pi / 180.0)
RAD2DEG: Final[np.float64] = np.float64(180.0 / np.pi)
EPS: Final[np.float64] = np.float64(1e-12)



def _as_1d_array(x: Union[AF, float]) -> AF64:
    """
    Convert scalars or array-like input into a flattened float64 array.
    
        - Scalars become length 1 array
        - Higher--dimensional inputs are flattened
        - dtype is normalized to float64
    
    Used to standardize inputs before vectorized or kernel operations.
    """
    if isinstance(x, (int, float)):
        return np.array([x], dtype=np.float64)
    
    array = np.asarray(x, dtype=np.float64)
    return array.ravel() if array.ndim > 0 \
        else np.array([array.item()], dtype=np.float64)



# ======================================================================
#       Cartesian to Spherical Conversion
# ======================================================================

@njit(fastmath=True, parallel=True, cache=True)
def _cartesian_to_spherical_kernel(
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    r: np.ndarray, p: np.ndarray, t: np.ndarray
) -> None:
    """
    Numba parallel kernel converting Cartesian coordinates to corresponding 
    spherical coordinates.
    """
    for i in prange(len(x)):
        xi = x[i]
        yi = y[i]
        zi = z[i]

        r2 = xi * xi + yi * yi
        z2 = zi * zi

        if r2 < EPS and z2 < EPS:
            r[i] = np.float64(0.0)
            p[i] = np.float64(0.0)
            t[i] = np.float64(0.0)
            continue

        ri = np.sqrt(r2 + z2)
        r[i] = ri
        p[i] = np.arctan2(yi, xi)

        ct = zi / ri
        ct = max(np.float64(-1.0), min(np.float64(1.0), ct))
        t[i] = np.arccos(ct)


def cartesian_to_spherical(
    dvec: Union[Af, Tuple[float, float, float]]
) -> Tuple[AF64, AF64, AF64]:
    """
    Convert Cartesian vectors to spherical coordinates. Always returns float64 
    arrays. Single vectors are internally reshaped to (1, 3) and all angles are
    measured in degrees.

    -----
    Args:
        dvec: Shape (3,) or (N, 3). Each row represented (x, y, z).
    --------
    Returns:
        (r, φ, θ)
            - r:    Radius 
            - φ:    Azimuth angle in degrees.
            - θ:    Inclination angle is degrees.
    -------
    Raises:
        ValueError: If input shape is invalid.
    """
    array = np.asarray(dvec, dtype=np.float64)
    if array.ndim == 0:
        raise ValueError("dvec has to be 1D or 2D")
    
    if array.ndim == 1:
        if array.size != 3:
            raise ValueError("dvec must have 3 elements when 1D")
        
        array = array.reshape(1, 3)
    
    elif array.ndim == 2:
        if array.shape[1] != 3:
            raise ValueError("dvec must have shape (N, 3)")
    
    else:
        raise ValueError("dvec must be either 1D or 2D")
    
    n = array.shape[0]
    x, y, z = array[:, 0], array[:, 1], array[:, 2]

    r = np.empty(n, dtype=np.float64)
    p = np.empty(n, dtype=np.float64)
    t = np.empty(n, dtype=np.float64)

    _cartesian_to_spherical_kernel(x, y, z, r, p, t)
    return r, p * RAD2DEG, t * RAD2DEG



# ======================================================================
#       Spherical to Cartesian Conversion
# ======================================================================

@njit(fastmath=True, parallel=True, cache=True)
def _spherical_to_cartesian_kernel(
    r: np.ndarray, p: np.ndarray, t: np.ndarray,
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> None:
    """
    Numba parallel kernel converting spherical coordinates to the corresponding
    Cartesian coordinates.
    """
    for i in prange(len(r)):
        ri = r[i]
        pi = p[i]
        ti = t[i]

        cp = np.cos(pi)
        sp = np.sin(pi)
        ct = np.cos(ti)
        st = np.sin(ti)

        ri_st = ri * st
        x[i] = ri_st * cp
        y[i] = ri_st * sp
        z[i] = ri * ct



def spherical_to_cartesian(
    radius: Union[AF, float], phi: Union[AF, float], theta: Union[AF, float]
) -> AF64:
    """
    Converts spherical coordinates to Cartesian coordinate vectors. Broadcasting 
    follows ``NumPy`` style semantics. Angle are computed in degrees.
    
    -----
    Args:
        radius: array-like
        phi: array-like (azimuth)
        theta: array-like (elevation)
    --------
    Returns:
        Array of shape (N, 3) containing (x, y, z)
    """
    # Convert to 1D arrays
    r, p, t = _as_1d_array(radius), _as_1d_array(phi), _as_1d_array(theta)
    n = max(r.size, p.size, t.size)

    # Broadcasting 
    if r.size == 1:
        r = np.full(n, r[0], dtype=np.float64)
    elif r.size != n:
        r = np.broadcast_to(r, (n,))
    
    if p.size == 1:
        p = np.full(n, p[0], dtype=np.float64)
    elif p.size != n:
        p = np.broadcast_to(p, (n,))
    
    if t.size == 1:
        t = np.full(n, t[0], dtype=np.float64)
    elif t.size != n:
        t = np.broadcast_to(t, (n,))
    

    p, t = p * DEG2RAD, t * DEG2RAD
    x = np.empty(n, dtype=np.float64)
    y = np.empty(n, dtype=np.float64)
    z = np.empty(n, dtype=np.float64)

    _spherical_to_cartesian_kernel(r, p, t, x, y, z)
    return np.column_stack((x, y, z))



# ======================================================================
#       Angle Combinations for Rotation
# ======================================================================

@njit(fastmath=True, parallel=True, cache=True)
def _angle_rotation_kernel(
    p0: np.ndarray, t0: np.ndarray,
    yaw: np.ndarray, pitch: np.ndarray,
    inv: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba parallel accelerated kernel for rotations
    """
    n = len(p0)
    p_out = np.empty(n, dtype=np.float64)
    t_out = np.empty(n, dtype=np.float64)

    for i in prange(n):
        # Spherical to Cartesian
        sp = np.sin(p0[i])
        cp = np.cos(p0[i])
        st = np.sin(t0[i])
        ct = np.cos(t0[i])

        x = st * cp
        y = st * sp
        z = ct

        # Apply rotation
        yw = yaw[i]
        pt = pitch[i]

        cy = np.cos(yw)
        sy = np.sin(yw)
        cpit = np.cos(pt)
        spit = np.sin(pt)

        if not inv:
            # Forward: R = Ry(pitch) @ Rz(yaw)
            # Rz(yaw)
            x1 = cy * x - sy * y
            y1 = sy * x + cy * y
            z1 = z

            # Ry(pitch)
            x2 = cpit * x1 + spit * z1
            y2 = y1
            z2 = -spit * x1 + cpit * z1
        
        else:
            # Inverse: R^-1 = Rz(-yaw) @ Ry(-pitch)
            # Ry(-pitch)
            x1 = cpit * x - spit * z
            y1 = y
            z1 = spit * x + cpit * z

            # Rz(-yaw)
            x2 = cy * x1 + sy * y1
            y2 = -sy * x1 + cy * y1
            z2 = z1
        
        # Normalize
        norm = np.sqrt(x2 * x2 + y2 * y2 + z2 * z2)
        if norm > 0.0:
            inv_norm = 1.0 / norm
            x2 *= inv_norm
            y2 *= inv_norm
            z2 *= inv_norm
        
        # Clamp for inverse trigonometry
        z2 = max(np.float64(-1.0), min(np.float64(1.0), z2))

        # Cartesian -> Spherical
        phi = np.arctan2(y2, x2)
        if phi < 0.0:
            phi += 2.0 * np.pi
        
        p_out[i] = phi
        t_out[i] = np.arccos(z2)
    
    return p_out, t_out



def _combine_angles(
    phi: AF64, theta: AF64, yaw: AF64, pitch: AF64, inv: bool = False
) -> Tuple[AF64, AF64]:
    """
    """
    # Ensure inputs are arrays
    phi = np.asarray(phi, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    yaw = np.asarray(yaw, dtype=np.float64)
    pitch = np.asarray(pitch, dtype=np.float64)

    # Broadcast all inputs to the same shape
    try:
        broadcasted_shape = np.broadcast_shapes(
            phi.shape, theta.shape, yaw.shape, pitch.shape
        )
    
    except ValueError as ve:
        raise ValueError(
            f"Cannot broadcast shapes: phi={phi.shape}, theta={theta.shape}, "
            f"yaw={yaw.shape}, pitch={pitch.shape}"
        ) from ve
    
    # Broadcast arrays to common shape (creates views when possible)
    phi = np.broadcast_to(phi, broadcasted_shape)
    theta = np.broadcast_to(theta, broadcasted_shape)
    yaw = np.broadcast_to(yaw, broadcasted_shape)
    pitch = np.broadcast_to(pitch, broadcasted_shape)

    # Convert to radians and flatten for kernel
    phi_r, theta_r = phi.ravel() * DEG2RAD, theta.ravel() * DEG2RAD
    yaw_r, pitch_r = yaw.ravel() * DEG2RAD, pitch.ravel() * DEG2RAD

    # Rotate
    phi_out_r, theta_out_r = _angle_rotation_kernel(
        phi_r, theta_r, yaw_r, pitch_r, inv
    )

    # Reshape and convert to degrees
    return phi_out_r.reshape(broadcasted_shape) * RAD2DEG, \
           theta_out_r.reshape(broadcasted_shape) * RAD2DEG



def add_angles(phi0: AF, theta0: AF, phi1: AF, theta1: AF) -> Tuple[AF64, AF64]:
    """
    Compose two spherical rotations. Returns the spherical angles obtained by
    applying (φ1, θ1) after (φ0, θ0). All angles are in degrees.
    """
    return _combine_angles(phi0, theta0, phi1, theta1, inv=False)


def sub_angles(phi0: AF, theta0: AF, phi1: AF, theta1: AF) -> Tuple[AF64, AF64]:
    """
    Subtract (invert) spherical rotation angles. Returns the spherical angles
    obtained by removing (φ1, θ1) from (φ0, θ0). All angles are in degrees.
    """
    return _combine_angles(phi0, theta0, phi1, theta1, inv=True)
