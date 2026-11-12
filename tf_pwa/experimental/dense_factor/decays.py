"""Amplitude basis per decay class.

Implements ``@basis`` from :mod:`.interface` for ``AmpDecay``,
``HelicityDecay``, ``DecayChain`` and ``DecayGroup``.  Importing this module
installs the basis; it imports ``particles`` first so the denominator terms a
basis may call are registered too.
"""

import itertools

import numpy as np
import tensorflow as tf

from tf_pwa.breit_wigner import Bprime_num
from tf_pwa.config import get_config
from tf_pwa.particle import cp_charge_group

from . import particles  # noqa: F401  (register the terms first)
from .interface import (
    AmpDecay,
    DecayChain,
    DecayGroup,
    HelicityDecay,
    basis,
    get_particle_model,
    rename_data_dict,
)

# ---------------------------------------------------------------------------
# Amplitude basis (numerator) per decay class
# ---------------------------------------------------------------------------


@basis("AmpDecay")
class _AmpDecayBasis:
    def get_amplitude_basis(self, *args, **kwargs):
        """Return the data-only basis of this decay."""
        raise NotImplementedError(
            "{} ({}) does not implement get_amplitude_basis".format(
                self, type(self).__name__
            )
        )

    def get_amplitude_coeff(self):
        """Return the current LS coefficients."""
        raise NotImplementedError(
            "{} ({}) does not implement get_amplitude_coeff".format(
                self, type(self).__name__
            )
        )


@basis("HelicityDecay")
class _HelicityBasis:
    def get_barrier_basis(self, q, mass, d=None):
        """Data-only part of q^l B_l'(q, q_0, d) = basis(q) * coeff(q_0)."""
        if d is None:
            d = self.d
        ls = self.get_l_list()
        l_min = min(ls) if self.force_min_l else None
        ret = []
        for l in ls:
            if l_min is not None:
                l = l_min
            b = tf.ones_like(q)
            if self.has_barrier_factor:
                if self.has_bprime:
                    b = b / Bprime_num(l, q, d)
                if self.has_ql:
                    b = b * q**l
            if self.add_covariant_term:
                if (
                    l - self.core.J + self.outs[0].J + self.outs[1].J
                ) % 2 == 1:
                    b = b * mass
            if self.barrier_factor_mass:
                b = b / mass**l
            ret.append(b)
        return tf.stack(ret, axis=-1)

    def get_barrier_coeff(self, q0, d=None, q02=None):
        """Parameter-only coefficient F_l(q_0) (and q_0^{-l} if norm)."""
        if d is None:
            d = self.d
        if q02 is None:
            q02 = q0 * q0
        if self.no_q0:
            q0 = tf.ones_like(q0)
            q02 = tf.ones_like(q02)
        ls = self.get_l_list()
        l_min = min(ls) if self.force_min_l else None
        ret = []
        for l in ls:
            if l_min is not None:
                l = l_min
            c = tf.cast(1.0, q0.dtype)
            if self.has_barrier_factor:
                if self.has_bprime:
                    c = Bprime_num(l, q0, d)
                if self.has_ql and self.barrier_factor_norm:
                    c = c / tf.cast(tf.abs(q02), c.dtype) ** (l / 2)
            ret.append(c)
        return tf.stack(ret)

    def get_amplitude_basis(self, data, data_p, **kwargs):
        if (
            type(self).get_amp is not HelicityDecay.get_amp
            or type(self).get_helicity_amp
            is not HelicityDecay.get_helicity_amp
            or type(self).get_ls_amp is not HelicityDecay.get_ls_amp
            or self.aligned
        ):
            return _AmpDecayBasis.get_amplitude_basis(
                self, data, data_p, **kwargs
            )
        basis = self.get_factor_angle_amp(data, data_p, **kwargs)
        if self.has_barrier_factor:
            q = self.cache_relative_p(data, data_p)
            bf = self.get_barrier_basis(q, data_p[self.core]["m"], self.d)
            bf = tf.cast(bf, basis.dtype)
            bf = tf.reshape(
                bf,
                [-1, bf.shape[1]] + [1] * (len(basis.shape) - 2),
            )
            basis = basis * bf
        return {"basis": basis, "transform": None}

    def get_amplitude_coeff(self):
        if (
            type(self).get_amp is not HelicityDecay.get_amp
            or type(self).get_helicity_amp
            is not HelicityDecay.get_helicity_amp
            or type(self).get_ls_amp is not HelicityDecay.get_ls_amp
            or self.aligned
        ):
            return _AmpDecayBasis.get_amplitude_coeff(self)
        if self.no_q0:
            q0 = tf.cast(1.0, get_config("dtype"))
            q02 = q0
        else:
            q0 = self.get_relative_momentum({}, False)
            q02 = self.get_relative_momentum2({}, False)
        c = self.get_barrier_coeff(q0, self.d, q02=q02)
        g = self.get_g_ls()
        coeff = g * tf.cast(c, g.dtype)
        return {"coeff": list(tf.unstack(coeff))}


@basis("DecayChain")
class _DecayChainBasis:
    def _check_dense_chain_support(self, method):
        if type(self).get_amp is not DecayChain.get_amp:
            raise NotImplementedError(
                "{} ({}) does not implement {}".format(
                    getattr(self, "chain", type(self).__name__),
                    type(self).__name__,
                    method,
                )
            )

    def get_amplitude_basis(
        self, data_c, data_p, all_data=None, base_map=None
    ):
        self._check_dense_chain_support("get_amplitude_basis")
        base_map = self.get_base_map(base_map)
        iter_idx = ["..."]
        amp_d = []
        indices = []
        next_map = "zyxwvutsr"
        used_idx = ""
        final_indices = self.amp_index(base_map)
        for i in self:
            tmp_idx = i.amp_index(base_map)
            tmp_idx = [next_map[0], *tmp_idx]
            indices.append(tmp_idx)
            used_idx += next_map[0]
            amp_d.append(
                i.get_amplitude_basis(data_c[i], data_p, all_data=all_data)[
                    "basis"
                ]
            )
            next_map = next_map[1:]
        final_indices = "".join(iter_idx + list(used_idx) + final_indices)

        if self.aligned:
            from tf_pwa.dfun import get_D_matrix_lambda

            for i in self:
                for idxj, j in enumerate(i.outs):
                    if j.J != 0:
                        ang = data_c[i][j].get("aligned_angle", None)
                        if ang is None and not getattr(
                            i, "helicity_inner_full", False
                        ):
                            continue
                        dt = get_D_matrix_lambda(
                            ang,
                            j.J,
                            i.list_helicity_inner()[idxj],
                            j.spins,
                        )
                        amp_d.append(tf.stop_gradient(dt))
                        idx = [base_map[j], base_map[j].upper()]
                        indices.append(idx)
                        final_indices = final_indices.replace(*idx)

        idxs = []
        for i in indices:
            tmp = "".join(iter_idx + i)
            idxs.append(tmp)
        idx = ",".join(idxs)
        idx_s = "{}->{}".format(idx, final_indices)
        return tf.einsum(idx_s, *amp_d)

    def get_amplitude_coeff(self):
        self._check_dense_chain_support("get_amplitude_coeff")
        return [i.get_amplitude_coeff()["coeff"] for i in self]

    def get_particle_terms(self, data_p, data_c, all_data=None):
        """Return ``[(particle, basis, coeff), ...]`` for internal lines."""
        self._check_dense_chain_support("get_particle_terms")
        ret = []
        if not self.inner:
            return ret
        for i in self.inner:
            if len(i.decay) >= 1:
                decay_i = i.decay[0]
                found = False
                for j in i.decay:
                    if j in self:
                        decay_i = j
                        found = True
                        break
                if not found:
                    raise IndexError(
                        "not found {} decay in {}".format(i, self)
                    )
                data_c_i = data_c[decay_i]
                if "|q|" not in data_c_i:
                    data_c_i["|q|"] = decay_i.get_relative_momentum(
                        data_p, True
                    )
                if "|q0|" not in data_c_i:
                    data_c_i["|q0|"] = decay_i.get_relative_momentum(
                        data_p, False
                    )
                if "|q|2" not in data_c_i:
                    data_c_i["|q|2"] = decay_i.get_relative_momentum2(
                        data_p, True
                    )
                if "|q0|2" not in data_c_i:
                    data_c_i["|q0|2"] = decay_i.get_relative_momentum2(
                        data_p, False
                    )
                basis = i.get_term_basis(
                    data_p[i], data_c_i, all_data=all_data
                )
            else:
                basis = i.get_term_basis(data_p[i], all_data=all_data)
            ret.append((i, basis, i.get_term_coeff()))
        return ret


@basis("DecayGroup")
class _DecayGroupBasis:
    def get_dense_structure(self):
        """Return the dense basis description, building it on first use.

        It follows from the chains and their term models only, so the data
        side can ask for it whatever amplitude model is loaded.  There is no
        separate init: the first caller builds it, everyone afterwards gets
        the same dict, and a failure propagates to that caller without
        leaving a half-built basis behind.
        """
        structure = getattr(self, "_dense_structure", None)
        if structure is None:
            structure = self._build_dense_structure()
            self._dense_structure = structure
        return structure

    def get_particle_terms(self, data):
        """Return per-chain particle term basis and coefficients."""
        from tf_pwa.amp.core import rename_data_dict

        data_particle = data["particle"]
        data_decay = data["decay"]

        used_chains = tuple([self.chains[i] for i in self.chains_idx])
        chain_maps = self.get_chains_map(used_chains)
        ret = []
        for decay_chain in used_chains:
            for chains in chain_maps:
                if str(decay_chain) in [str(i) for i in chains]:
                    maps = chains[decay_chain]
                    break
            chain_topo = decay_chain.standard_topology()
            found = False
            for i in data_decay.keys():
                if i == chain_topo:
                    data_decay_i = data_decay[i]
                    found = True
                    break
            if not found:
                raise KeyError("not found {}".format(chain_topo))
            data_c = rename_data_dict(data_decay_i, maps)
            data_p = rename_data_dict(data_particle, maps)
            ret.append(
                decay_chain.get_particle_terms(data_p, data_c, all_data=data)
            )
        return ret

    def get_amplitude_basis(self, data):
        from tf_pwa.amp.core import rename_data_dict

        data_particle = data["particle"]
        data_decay = data["decay"]

        used_chains = tuple([self.chains[i] for i in self.chains_idx])
        chain_maps = self.get_chains_map(used_chains)
        base_map = self.get_base_map()
        ret = []
        for decay_chain in used_chains:
            for chains in chain_maps:
                if str(decay_chain) in [str(i) for i in chains]:
                    maps = chains[decay_chain]
                    break
            chain_topo = decay_chain.standard_topology()
            found = False
            for i in data_decay.keys():
                if i == chain_topo:
                    data_decay_i = data_decay[i]
                    found = True
                    break
            if not found:
                raise KeyError("not found {}".format(chain_topo))
            data_c = rename_data_dict(data_decay_i, maps)
            data_p = rename_data_dict(data_particle, maps)
            amp = decay_chain.get_amplitude_basis(
                data_c, data_p, base_map=base_map, all_data=data
            )
            ret.append(amp)
        return ret

    def get_amplitude_coeff(self):
        return [self.chains[i].get_amplitude_coeff() for i in self.chains_idx]

    def _build_dense_structure(self):
        """Build the dense basis description; called once, from the getter.

        Everything here follows from the chains and their term models, so it
        is the group's data, shared by every model over this group.  Only the
        first dense model (or the preprocessor) pays for it, and an
        unsupported term model is rejected there rather than on behalf of a
        stock model that never asks.
        """
        chains = list(self.chains)

        u_chain = []
        u_ls = []
        u_term = []
        # the chains themselves stay on the decay group; only the layout of
        # the denominator-coefficient vector is stored
        den_sizes = []
        param_offset = []
        max_internal_lines = 0

        off = 0
        for k, chain in enumerate(chains):
            decays = list(chain)
            lines = list(chain.inner)
            n_ls = [len(d.get_ls_list()) for d in decays]
            # a line whose model has no rational basis makes the whole
            # model unsupported, so fail here with the term's own message
            # instead of leaving a husk that raises "not initialized" only
            # when the first batch arrives
            terms = [p.get_term_coeff() for p in lines]
            n_term = [len(t["num"]) for t in terms]
            sizes = [len(t["den"]) for t in terms]
            offsets = []
            for size in sizes:
                offsets.append(off)
                off += size
            den_sizes.append(sizes)
            param_offset.append(offsets)

            max_internal_lines = max(max_internal_lines, len(lines))

            ls_combos = (
                list(itertools.product(*[range(i) for i in n_ls]))
                if n_ls
                else [()]
            )
            term_combos = (
                list(itertools.product(*[range(i) for i in n_term]))
                if n_term
                else [()]
            )
            for ls in ls_combos:
                for term in term_combos:
                    u_chain.append(k)
                    u_ls.append(list(ls))
                    u_term.append(list(term))

        structure = {
            "den_sizes": den_sizes,
            "param_offset": param_offset,
            "u_chain": u_chain,
            "u_ls": u_ls,
            "u_term": u_term,
            "n_param": off,
            "n_u": len(u_chain),
            "max_internal_lines": max_internal_lines,
            # the helicity axes are a particle property, one per top/final,
            # so they are known here rather than when the first F is built
            "f_shape": (len(self.top.spins),)
            + tuple(len(p.spins) for p in self.outs),
            # likewise the CP transform, which only walks the group
            "cp": self._cp_transform(),
        }
        # the basis transform of B; kept next to the terms that apply it
        self.M2 = tf.eye(off, dtype=tf.float64)
        return structure

    def swap_terms(self, F, key):
        """Sign of one swap's ``F`` block, in the shared axis convention.

        ``get_amp2`` adds ``factor * transpose(amp_swapped)``.  The transpose
        is applied to the *amplitude* (see ``_swap_amplitude``), not to the
        basis: every block is built in the same column convention, so one
        compression record describes all of them.  Transposing ``F`` here
        instead would rotate the stored columns out of that convention and
        break any per-column transform (the fused op applies the record to
        the block as stored).
        """
        factor = tf.cast(float(self.get_swap_factor(key)), F.dtype)
        return F * factor

    def _swap_amplitude(self, amp, key):
        """``transpose(amp)`` over the particle axes, as ``get_amp2`` does."""
        f_shape = tuple(self.get_dense_structure()["f_shape"])
        n = tf.shape(amp)[0]
        out = tf.reshape(amp, (n,) + f_shape)
        out = tf.transpose(
            out, self.get_id_swap_transpose(key, len(f_shape) + 1)
        )
        return tf.reshape(out, (n, tf.shape(amp)[1]))

    def build_dense_terms(self, data):
        structure = self.get_dense_structure()
        old_chains_idx = self.chains_idx
        self.chains_idx = list(range(len(self.chains)))
        try:
            F_chains = self.get_amplitude_basis(data)
            particle_terms = self.get_particle_terms(data)
        finally:
            self.chains_idx = old_chains_idx

        F_list = []
        B_blocks = []
        n_data = tf.shape(F_chains[0])[0]

        for idx, chain in enumerate(self.chains):
            F_chain = F_chains[idx]
            decays = list(chain)
            lines = list(chain.inner)
            n_ls = [len(d.get_ls_list()) for d in decays]
            n_term = [len(term[1]["num"]) for term in particle_terms[idx]]

            n_ls_combo = int(np.prod(n_ls)) if n_ls else 1
            n_decay = len(decays)
            f_shape = list(F_chain.shape[1 + n_decay :])
            if any(i is None for i in f_shape):
                raise NotImplementedError(
                    "dense_factor needs a static helicity shape"
                )
            shape_t = tuple(int(i) for i in f_shape)
            if structure["f_shape"] != shape_t:
                raise NotImplementedError(
                    "dense_factor needs one helicity shape for every chain"
                )
            if len(shape_t) != 1 + len(self.outs):
                raise NotImplementedError(
                    "dense_factor needs one helicity axis per particle "
                    "({} for {})".format(1 + len(self.outs), shape_t)
                )
            F_ls = tf.reshape(F_chain, [n_data, n_ls_combo] + f_shape)

            term_combos = (
                list(itertools.product(*[range(i) for i in n_term]))
                if n_term
                else [()]
            )
            F_num = []
            for combo in term_combos:
                f = tf.ones((n_data,), dtype=F_chain.dtype)
                for l, t in enumerate(combo):
                    f = f * tf.cast(
                        particle_terms[idx][l][1]["num"][t],
                        F_chain.dtype,
                    )
                F_num.append(f)

            for ls_idx in range(n_ls_combo):
                base = F_ls[:, ls_idx]
                shape = [n_data] + [1] * (len(base.shape) - 1)
                for f in F_num:
                    F_list.append(base * tf.reshape(f, shape))

            for l in range(len(lines)):
                B_blocks.append(particle_terms[idx][l][1]["den"])

        if F_list:
            F = tf.stack(F_list, axis=1)
            F = tf.reshape(F, [n_data, len(F_list), -1])
            F = tf.transpose(F, [0, 2, 1])
        else:
            F = tf.zeros((n_data, 0, 0), dtype=F_chains[0].dtype)
        B_cols = [
            tf.cast(col, F_chains[0].dtype)
            for block in B_blocks
            for col in block
        ]
        if B_cols:
            B = tf.stack(B_cols, axis=-1)
            B = tf.matmul(B, tf.cast(self.M2, B.dtype), transpose_b=True)
        else:
            B = tf.zeros((n_data, 0), dtype=F_chains[0].dtype)
        dtype = get_config("complex_dtype")
        # Event-first: (n_data, n_param).  Batching slices the event axis
        # contiguously and the fused kernel reads one event's basis block
        # without a transpose.  The sparse fallback below transposes it;
        # that cost belongs to the fallback, not to the stored layout.
        return tf.cast(F, dtype), tf.cast(B, dtype)

    def _cp_transform(self):
        """``(frac, change)`` for the CP-conjugate amplitude, like get_amp3.

        ``frac`` is the product of the C parities of the particles that the
        CP map leaves in place; ``change`` is the particle permutation of the
        ones that move.
        """
        if not getattr(self, "cp_particles", None):
            return 1.0, ()
        name_map = {str(i): i for i in self.outs}
        frac = 1.0
        change = []
        for a, b in cp_charge_group(
            [str(i) for i in self.outs],
            self.identical_particles,
            self.cp_particles,
        ):
            for i, j in zip(a, b):
                if i == j:
                    frac = frac * getattr(name_map[i], "C", -1)
                else:
                    change.append((i, j))
        return float(frac), tuple(change)

    def build_dense_coeff(self, structure):
        """Build numerator and denominator coefficients for a term cache."""
        C = []

        chain_c = []
        ls_vecs = []
        num_vecs = []
        den_vecs = []
        for chain in self.chains:
            decays = list(chain)
            lines = list(chain.inner)
            chain_c.append(tf.reshape(chain.get_amp_total(), (-1,))[0])
            ls_vecs.append([d.get_amplitude_coeff()["coeff"] for d in decays])
            term_coeffs = [p.get_term_coeff() for p in lines]
            num_vecs.append([c["num"] for c in term_coeffs])
            den_vecs.append([c["den"] for c in term_coeffs])

        dtype = get_config("complex_dtype")
        for u, k in enumerate(structure["u_chain"]):
            c = tf.cast(chain_c[k], dtype)
            for d, ls_idx in enumerate(structure["u_ls"][u]):
                c = c * tf.cast(ls_vecs[k][d][ls_idx], dtype)
            for l, term_idx in enumerate(structure["u_term"][u]):
                c = c * tf.cast(num_vecs[k][l][term_idx], dtype)
            C.append(tf.cast(c, dtype))

        g_blocks = [g for row in den_vecs for block in row for g in block]
        if C:
            C = tf.stack(C)
        else:
            C = tf.zeros((0,), dtype=dtype)
        # the denominator coefficients are real; only C is complex
        rtype = get_config("dtype")
        if g_blocks:
            g = tf.cast(tf.stack(g_blocks), rtype)
        else:
            g = tf.zeros((0,), dtype=rtype)
        return C, g
