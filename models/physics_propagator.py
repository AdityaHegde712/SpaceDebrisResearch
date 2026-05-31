"""
Physics Orbit Propagator (Keplerian + J2)

Provides two integration methods:
  1. DOP853 (adaptive, high-accuracy) via scipy.solve_ivp
  2. Fixed-step RK4 (fast, adequate for short/medium horizons)

Usage:
    from models.physics_propagator import (
        propagate_epochs, propagate_single,
        propagate_fixed_rk4, propagate_fixed_epochs,
    )
"""

import numpy as np
from scipy.integrate import solve_ivp

# Constants (WGS-84)
GM = 398600.4418       # km³/s²
J2 = 1.08263e-3        # dimensionless
R_EARTH = 6378.137      # km


# =========================================================================
# Common acceleration model
# =========================================================================

def two_body_j2_acceleration(t, state):
    """
    Compute state derivatives for two-body + J2.
    Used as the ODE RHS function for scipy.integrate.solve_ivp.

    Parameters
    ----------
    t : float
        Time (unused, needed for scipy interface)
    state : array_like, shape (6,)
        [x, y, z, vx, vy, vz] in J2000 inertial frame

    Returns
    -------
    dstate_dt : ndarray, shape (6,)
        [vx, vy, vz, ax, ay, az]
    """
    x, y, z, vx, vy, vz = state
    r = np.array([x, y, z])
    r_mag = np.linalg.norm(r)
    r3 = r_mag ** 3

    # Two-body acceleration
    ax = -GM * x / r3
    ay = -GM * y / r3
    az = -GM * z / r3

    # J2 perturbation
    r2 = r_mag ** 2
    factor = -1.5 * J2 * GM * R_EARTH ** 2 / (r2 ** 2.5)
    ax += factor * x * (1 - 5 * z**2 / r2)
    ay += factor * y * (1 - 5 * z**2 / r2)
    az += factor * z * (3 - 5 * z**2 / r2)

    return np.array([vx, vy, vz, ax, ay, az])


def propagate_epochs(initial_state, epoch_times_s, rtol=1e-10, atol=1e-12, max_step=60.0):
    """
    Propagate an initial state to a set of target epoch times using DOP853.

    Parameters
    ----------
    initial_state : ndarray, shape (6,)
        [x, y, z, vx, vy, vz] at t=0
    epoch_times_s : ndarray, shape (n_epochs,)
        Target times in seconds (positive, monotonically increasing)
    rtol : float
        Relative tolerance for adaptive integrator
    atol : float
        Absolute tolerance for adaptive integrator
    max_step : float
        Maximum step size (seconds). 60s default for LEO accuracy.

    Returns
    -------
    states : ndarray, shape (n_epochs, 6)
        Propagated states at each target epoch time
    """
    epoch_times_s = np.asarray(epoch_times_s, dtype=float)
    if len(epoch_times_s) == 0:
        return initial_state.reshape(1, 6)

    t_span = (0.0, epoch_times_s[-1])
    t_eval = epoch_times_s

    sol = solve_ivp(
        two_body_j2_acceleration,
        t_span,
        initial_state,
        method='DOP853',
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )

    if not sol.success:
        raise RuntimeError(f"Propagation failed: {sol.message}")

    # Transpose from (6, n_epochs) to (n_epochs, 6)
    return sol.y.T


def propagate_single(initial_state, target_seconds, rtol=1e-10, atol=1e-12, max_step=60.0):
    """
    Propagate an initial state to a single target time.

    Parameters
    ----------
    initial_state : ndarray, shape (6,)
        [x, y, z, vx, vy, vz] at t=0
    target_seconds : float
        Target propagation time in seconds
    rtol, atol : float
        Tolerances for adaptive integrator
    max_step : float
        Maximum step size (seconds)

    Returns
    -------
    final_state : ndarray, shape (6,)
        Propagated state at target time
    """
    states = propagate_epochs(initial_state, np.array([target_seconds]),
                              rtol=rtol, atol=atol, max_step=max_step)
    return states[0]


# Backward compatibility alias
propagate_to_horizon = propagate_single


# =========================================================================
# Fixed-step RK4 propagator (fast, for bulk propagation)
# =========================================================================

def _force_fixed(state):
    """Acceleration (two-body + J2).  Returns [vx,vy,vz,ax,ay,az]."""
    x, y, z, vx, vy, vz = state
    r = np.array([x, y, z])
    r_mag = np.linalg.norm(r)
    r3 = r_mag ** 3
    ax = -GM * x / r3
    ay = -GM * y / r3
    az = -GM * z / r3
    r2 = r_mag ** 2
    factor = -1.5 * J2 * GM * R_EARTH ** 2 / (r2 ** 2.5)
    ax += factor * x * (1 - 5 * z**2 / r2)
    ay += factor * y * (1 - 5 * z**2 / r2)
    az += factor * z * (3 - 5 * z**2 / r2)
    return np.array([vx, vy, vz, ax, ay, az])


def _rk4_step(state, dt):
    """Single RK4 step (returns new state)."""
    k1 = _force_fixed(state)
    k2 = _force_fixed(state + 0.5 * dt * k1)
    k3 = _force_fixed(state + 0.5 * dt * k2)
    k4 = _force_fixed(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def propagate_fixed_rk4(initial_state, target_seconds, dt=60.0):
    """
    Propagate to a single target time using fixed-step RK4.

    Parameters
    ----------
    initial_state : ndarray (6,)
        [x, y, z, vx, vy, vz] at t=0
    target_seconds : float
        Target propagation time (seconds)
    dt : float
        Step size (seconds).  60s recommended for LEO (95 steps/orbit).

    Returns
    -------
    final_state : ndarray (6,)
        Propagated state at target time
    """
    n_steps = max(1, int(target_seconds / dt))
    actual_dt = target_seconds / n_steps
    state = initial_state.copy()
    for _ in range(n_steps):
        state = _rk4_step(state, actual_dt)
    return state


def propagate_fixed_epochs(initial_state, epoch_times_s, dt=60.0):
    """
    Propagate to multiple epoch times, propagating segment by segment.

    Each segment goes from previous epoch to next epoch, so we avoid
    re-propagating overlapping intervals.  Total RK4 steps = total_time/dt.

    Parameters
    ----------
    initial_state : ndarray (6,)
        [x, y, z, vx, vy, vz] at t=0
    epoch_times_s : array_like
        Target times in seconds (monotonically increasing, positive)
    dt : float
        Step size (seconds)

    Returns
    -------
    states : ndarray (n_epochs, 6)
        Propagated states at each target epoch time
    """
    epoch_times_s = np.asarray(epoch_times_s, dtype=float)
    if len(epoch_times_s) == 0:
        return initial_state.reshape(1, 6)

    states = [initial_state.copy()]
    last_t = 0.0

    for epoch_t in epoch_times_s[1:]:
        delta = epoch_t - last_t
        seg_state = propagate_fixed_rk4(states[-1], delta, dt=dt)
        states.append(seg_state)
        last_t = epoch_t

    return np.array(states)


def verify_propagator():
    """
    Quick verification: propagate a circular orbit for 1 orbit.
    Without J2: should return to starting position (< 1m error).
    With J2-only (via flag): should show small precession.
    """
    # Circular orbit at 500 km altitude
    R = R_EARTH + 500.0
    v_circ = np.sqrt(GM / R)
    state0 = np.array([R, 0.0, 0.0, 0.0, v_circ, 0.0])
    T = 2 * np.pi * np.sqrt(R**3 / GM)

    print(f"Verification: 500 km circular orbit, period = {T:.1f}s")

    states = propagate_epochs(state0, np.array([T]), rtol=1e-12, atol=1e-14, max_step=60.0)
    pos_err = np.linalg.norm(states[0, :3] - state0[:3])
    vel_err = np.linalg.norm(states[0, 3:] - state0[3:])
    print(f"  1 orbit (J2 ON):  pos error = {pos_err:.6f} km, vel error = {vel_err:.8f} km/s")
    print(f"  Altitude: {np.linalg.norm(states[0, :3]) - R_EARTH:.3f} km")

    # Verify energy conservation
    r = np.linalg.norm(states[0, :3])
    v = np.linalg.norm(states[0, 3:])
    E_final = v**2 / 2 - GM / r
    E_init = v_circ**2 / 2 - GM / R
    print(f"  Energy error: {abs(E_final - E_init):.2e} (rel: {abs(E_final - E_init) / abs(E_init):.2e})")

    return pos_err, vel_err


if __name__ == "__main__":
    verify_propagator()
