import numpy as np

# Minkowski metric (1, -1, -1, -1)
METRIC = np.array([1.0, -1.0, -1.0, -1.0])


def mass_sq(p4):
    """Compute invariant mass squared: m^2 = E^2 - px^2 - py^2 - pz^2"""
    return np.sum(p4 * p4 * METRIC, axis=-1)


class MassCons:
    """
    Kinematic constraint model for particle physics.

    Always constrains individual particle masses.
    Optionally adds extra mass constraints (e.g., total mass).
    """

    def __init__(self, n_particles, mass_constraints=None):
        """
        Parameters:
        -----------
        n_particles : int
            Number of particles in the fit
        mass_constraints : list of tuples or None, optional
            Extra mass constraints beyond individual particles.
            Each tuple contains particle indices to combine for a mass constraint.
            Example: [(0,1,2)] adds total mass constraint on particles 0, 1, 2.
            If None, only individual particle masses are constrained.
            Note: If momentum_target is used in do_constraints(), avoid adding
            total mass constraint here as it's already implied by fixed 4-momentum.
        """
        self.n_particles = n_particles
        self.n_mass_constraints = n_particles
        if mass_constraints:
            self.n_mass_constraints += len(mass_constraints)
        self.n_dim = n_particles * 4

        # Pre-compute indices: individual particles + extra constraints
        self._mass_indices_list = [
            np.array([i], dtype=np.int32) for i in range(n_particles)
        ]
        if mass_constraints:
            for c in mass_constraints:
                self._mass_indices_list.append(np.array(c, dtype=np.int32))

    def do_constraints(
        self,
        x,
        mass_targets_sq=None,
        momentum_target=None,
        max_iter=50,
        tol=1e-12,
        chunk_size=2000,
    ):
        """
        Apply kinematic constraints using Newton-Raphson iteration.

        Parameters:
        -----------
        x : ndarray
            Input 4-momenta, shape (n_events, n_particles, 4)
        mass_targets_sq : ndarray or None, optional
            Target mass-squared values, shape (n_mass_constraints,) or (n_events, n_mass_constraints).
            If None, computed from the mean of input data for each constraint
            (individual particles and combined masses). Input data should already
            be close to physical values for meaningful results.
        momentum_target : ndarray or None, optional
            Target total 4-momentum [E, px, py, pz], shape (4,) or (n_events, 4).
            If None, no momentum constraint is applied.
        max_iter : int
            Maximum number of iterations
        tol : float
            Convergence tolerance
        chunk_size : int
            Number of events per chunk. Default: 2000.

        Returns:
        --------
        ndarray : Adjusted 4-momenta
        """
        n_events = x.shape[0]
        n_mass = self.n_mass_constraints

        if mass_targets_sq is None:
            mass_targets_sq = []
            for idx in self._mass_indices_list:
                total_p = np.sum(x[:, idx], axis=1)
                mass_targets_sq.append(np.mean(mass_sq(total_p)))
            mass_targets_sq = np.stack(mass_targets_sq, axis=-1)

        # Broadcast targets upfront
        mass_arr = self._broadcast_to_events(mass_targets_sq, n_events, n_mass)
        mom_arr = None
        if momentum_target is not None:
            mom_arr = self._broadcast_to_events(momentum_target, n_events, 4)

        # No chunking needed for small datasets
        if chunk_size >= n_events:
            return self._do_chunk(x, mass_arr, mom_arr, max_iter, tol)

        # Process in chunks
        result = np.empty_like(x)
        n_chunks = (n_events + chunk_size - 1) // chunk_size

        for i in range(n_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, n_events)
            chunk_mom = mom_arr[start:end] if mom_arr is not None else None
            result[start:end] = self._do_chunk(
                x[start:end], mass_arr[start:end], chunk_mom, max_iter, tol
            )

        return result

    def _broadcast_to_events(self, arr, n_events, last_dim):
        """Broadcast array to (n_events, last_dim) shape."""
        arr = np.asarray(arr)
        if arr.ndim == 1:
            return np.broadcast_to(arr, (n_events, last_dim)).copy()
        elif arr.shape[0] == 1:
            return np.broadcast_to(arr[0], (n_events, last_dim)).copy()
        return arr

    def _do_chunk(self, x, mass_targets_sq, momentum_target, max_iter, tol):
        """Process a single chunk of events. Arrays already broadcast to (n_events, dim)."""
        n_events = x.shape[0]
        n_particles = self.n_particles
        n_mass = self.n_mass_constraints
        n_dim = self.n_dim

        if momentum_target is not None:
            n_constraints = n_mass + 4
        else:
            n_constraints = n_mass

        z = x.copy().reshape(n_events, n_dim)

        # Pre-allocate arrays
        J = np.zeros((n_events, n_constraints, n_dim))
        c_vals = np.zeros((n_events, n_constraints))

        eye_diag = np.eye(n_constraints) * 1e-12
        mass_indices_list = self._mass_indices_list

        z_p = z.reshape(n_events, n_particles, 4)

        for iteration in range(max_iter):
            # === Mass constraints ===
            for c_idx in range(n_mass):
                idx = mass_indices_list[c_idx]
                total_p = np.sum(z_p[:, idx], axis=1)
                c_vals[:, c_idx] = mass_sq(total_p) - mass_targets_sq[:, c_idx]
                grad = 2.0 * total_p * METRIC
                for pid in idx:
                    J[:, c_idx, 4 * pid : 4 * pid + 4] = grad

            # Total momentum
            total_p_all = np.sum(z_p, axis=1)

            # === Momentum constraints ===
            if momentum_target is not None:
                c_vals[:, n_mass:] = total_p_all - momentum_target
                for comp in range(4):
                    J[:, n_mass + comp, comp::4] = 1.0

            # Check convergence
            if np.max(np.abs(c_vals)) < tol:
                break

            # Newton step
            JJT = np.matmul(J, J.transpose(0, 2, 1))
            JJT += eye_diag
            lambda_vals = np.linalg.solve(
                JJT, c_vals[:, :, np.newaxis]
            ).squeeze(-1)
            z -= np.matmul(
                J.transpose(0, 2, 1), lambda_vals[:, :, np.newaxis]
            ).squeeze(-1)
            z_p = z.reshape(n_events, n_particles, 4)

        return z.reshape(n_events, n_particles, 4)
