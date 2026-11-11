#include <cuda_runtime.h>

#define NWARP 8
#define SHMEM_LIMIT 49152   /* default dynamic shared memory per block */
#define MAXUPW 4            /* chains per lane kept in registers (nu <= 128) */

// den[u,e] = prod_l sum_i g[off[u*nline+l]+i] B[e, off[u*nline+l]+i]
// A_p[e]   = sum_u F[p,e,u] * C[u] / den[u,e]
//
// One warp owns one event: it computes W[u] = C[u]/den[u,e] once, then loops
// the projections with a scalar accumulator, so the projection count is a
// runtime value and costs no per-projection register state.
//
// c is complex, g is real: every term in the plugin builds real denominator
// coefficients, so the kernels take g as doubles and return a real dg.  A
// complex g would only add an always-zero imaginary product.
//
// W is kept in registers as MAXUPW values per lane (u = lane + 32*k) while
// the chains fit, i.e. nu <= 32*MAXUPW = 128, and in a per-warp shared slice
// (wpb*nu*double2) beyond that.  The register variant avoids re-reading W
// from shared memory for every projection; dispatch picks it from nu.

// ``g`` is the denominator coefficient vector and is real by construction
// for every term in the plugin: the mass, the width and the Flatte/GS
// couplings are real, and the complex content lives in ``c`` and in the
// spline numerator.
template <int S>
__device__ __forceinline__ void line_den(
    const double2* __restrict__ B, const double* __restrict__ G,
    long long bo, int o, int s, double& dr, double& di)
{
    dr = 0.0; di = 0.0;
    #pragma unroll
    for (int i = 0; i < S; i++) {
        if (i < s) {
            const double2 b = B[bo+o+i];
            const double g = G[o+i];
            dr += g*b.x;
            di += g*b.y;
        }
    }
}

// Same sum when the basis columns are stored merged: column i lives at
// MI[o+i] in the reduced basis, so no expanded copy is needed.
template <int S>
__device__ __forceinline__ void line_den_merge(
    const double2* __restrict__ B, const int* __restrict__ MI,
    const double* __restrict__ G,
    long long bo, int o, int s, double& dr, double& di)
{
    dr = 0.0; di = 0.0;
    #pragma unroll
    for (int i = 0; i < S; i++) {
        if (i < s) {
            const double2 b = B[bo + MI[o+i]];
            const double g = G[o+i];
            dr += g*b.x;
            di += g*b.y;
        }
    }
}

__device__ __forceinline__ double2 warp_sum2(double ar, double ai)
{
    for (int o = 16; o > 0; o >>= 1) {
        ar += __shfl_down_sync(0xffffffffu, ar, o);
        ai += __shfl_down_sync(0xffffffffu, ai, o);
    }
    return make_double2(__shfl_sync(0xffffffffu, ar, 0),
                        __shfl_sync(0xffffffffu, ai, 0));
}

template <int S, bool REG>
__global__ void dense_fwd_kernel(
    const double2* __restrict__ F,   // [ne * np * nu]
    const double2* __restrict__ B,
    const double2* __restrict__ C,
    const double* __restrict__ G,
    const int* __restrict__ off, const int* __restrict__ sz,
    const int* __restrict__ MU, const int* __restrict__ MI,
    int ne, int nu, int nu_red, int nparam, int nparam_red, int np,
    int merged,
    double2* __restrict__ A_out)     // [ne * np], the amplitude
{
    extern __shared__ double2 wsh[];
    const int lane  = threadIdx.x & 31;
    const int wid   = threadIdx.x >> 5;
    const int wpb   = blockDim.x >> 5;
    const int gwarp = blockIdx.x * wpb + wid;
    const int nwarp = gridDim.x * wpb;
    double2* Wl = wsh + (size_t)wid * nu;

    for (int e = gwarp; e < ne; e += nwarp) {
        const long long bo = (long long)e * nparam_red;
        const long long fo = (long long)e * (long long)np * nu_red;
        if constexpr (REG) {
            double2 wc[MAXUPW];
            int fu[MAXUPW];
            #pragma unroll
            for (int k = 0; k < MAXUPW; k++) {
                const int u = lane + 32*k;
                if (u < nu) {
                    fu[k] = merged ? MU[u] : u;
                    const int o = off[u], s = sz[u];
                    double dr = 1.0, di = 0.0;
                    if (s > 0) {
                        if (merged) line_den_merge<S>(B, MI, G, bo, o, s, dr, di);
                        else line_den<S>(B, G, bo, o, s, dr, di);
                    }
                    const double den = dr*dr + di*di;
                    const double2 c = C[u];
                    wc[k] = make_double2((c.x*dr + c.y*di) / den,
                                         (c.y*dr - c.x*di) / den);
                }
            }
            for (int p = 0; p < np; p++) {
                double ar = 0.0, ai = 0.0;
                #pragma unroll
                for (int k = 0; k < MAXUPW; k++) {
                    const int u = lane + 32*k;
                    if (u < nu) {
                        const double2 f =
                            F[fo + (long long)p*nu_red + fu[k]];
                        ar += f.x*wc[k].x - f.y*wc[k].y;
                        ai += f.x*wc[k].y + f.y*wc[k].x;
                    }
                }
                const double2 A = warp_sum2(ar, ai);
                if (lane == 0) A_out[(long long)e*np + p] = A;
            }
        } else {
            for (int u = lane; u < nu; u += 32) {
                const int o = off[u], s = sz[u];
                double dr = 1.0, di = 0.0;
                if (s > 0) {
                    if (merged) line_den_merge<S>(B, MI, G, bo, o, s, dr, di);
                    else line_den<S>(B, G, bo, o, s, dr, di);
                }
                const double den = dr*dr + di*di;
                const double2 c = C[u];
                Wl[u] = make_double2((c.x*dr + c.y*di) / den,
                                     (c.y*dr - c.x*di) / den);
            }
            __syncwarp();
            for (int p = 0; p < np; p++) {
                double ar = 0.0, ai = 0.0;
                for (int u = lane; u < nu; u += 32) {
                    const int fu = merged ? MU[u] : u;
                    const double2 f = F[fo + (long long)p*nu_red + fu];
                    const double2 wv = Wl[u];
                    ar += f.x*wv.x - f.y*wv.y;
                    ai += f.x*wv.y + f.y*wv.x;
                }
                const double2 A = warp_sum2(ar, ai);
                if (lane == 0) A_out[(long long)e*np + p] = A;
            }
        }

    }
}

template <int S>
__global__ void dense_grad_kernel(
    const double2* __restrict__ F,
    const double2* __restrict__ B,
    const double2* __restrict__ C,
    const double* __restrict__ G,
    const int* __restrict__ off, const int* __restrict__ sz,
    const int* __restrict__ MU, const int* __restrict__ MI,
    const double* __restrict__ dla,  // dL/dA, [ne * np] complex
    int ne, int nu, int nu_red, int nparam, int nparam_red, int np,
    int merged,
    double* __restrict__ dc_part, double* __restrict__ dg_part)
{
    const double2* DLA = reinterpret_cast<const double2*>(dla);
    for (int u = threadIdx.x; u < nu; u += blockDim.x) {
        const int o = off[u], s = sz[u];
        const int fu = merged ? MU[u] : u;
        const double2 c = C[u];
        double dcr = 0.0, dci = 0.0;
        double dgr[S];
        #pragma unroll
        for (int i = 0; i < S; i++) { dgr[i] = 0.0; }

        for (int e = blockIdx.x; e < ne; e += gridDim.x) {
            const long long bo = (long long)e * nparam_red;
            const long long fo = (long long)e * (long long)np * nu_red;
            const long long ao = (long long)e * np;
            double2 bv[S];
            #pragma unroll
            for (int i = 0; i < S; i++)
                if (i < s) bv[i] = B[bo + (merged ? MI[o+i] : o+i)];
            double dr = 1.0, di = 0.0;
            if (s > 0) {
                if (merged) line_den_merge<S>(B, MI, G, bo, o, s, dr, di);
                else line_den<S>(B, G, bo, o, s, dr, di);
            }
            double gwr = 0.0, gwi = 0.0;
            for (int p = 0; p < np; p++) {
                const double2 da = DLA[ao+p];
                const double2 f = F[fo + (long long)p*nu_red + fu];
                gwr += da.x*f.x + da.y*f.y;
                gwi += da.y*f.x - da.x*f.y;
            }
            const double den = dr*dr + di*di;
            const double icr = dr/den, ici = di/den;
            dcr += gwr*icr - gwi*ici;
            dci += gwr*ici + gwi*icr;
            const double gir = c.x*gwr + c.y*gwi;
            const double gii = c.x*gwi - c.y*gwr;
            const double pr = dr*dr - di*di, pi = -2.0*dr*di;
            const double pn = pr*pr + pi*pi;
            const double ipr =  pr/pn, ipi = -pi/pn;
            const double gdr = -(gir*ipr - gii*ipi);
            const double gdi = -(gir*ipi + gii*ipr);
            #pragma unroll
            for (int i = 0; i < S; i++) {
                if (i < s) {
                    dgr[i] += gdr*bv[i].x + gdi*bv[i].y;
                }
            }
        }

        dc_part[(long long)blockIdx.x*2*nu + 2*u]     = dcr;
        dc_part[(long long)blockIdx.x*2*nu + 2*u + 1] = dci;
        #pragma unroll
        for (int i = 0; i < S; i++) {
            if (i < s) {
                atomicAdd(&dg_part[(long long)blockIdx.x*nparam + (o+i)],
                          dgr[i]);
            }
        }
    }
}

__global__ void reduce_k(const double* __restrict__ part, int nb, int row, int off,
                         int m, double* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    double s = 0.0;
    for (int b = 0; b < nb; b++) s += part[(long long)b*row + off + i];
    out[i] = s;
}

// ---------------------------------------------------------------------------
// General case: any number of denominator lines per chain.
// The forward keeps den in D_out[e,u]; the gradient needs one factor per
// line and D/that factor.
// ---------------------------------------------------------------------------

template <int S, bool REG>
__global__ void dense_fwd_kernel_n(
    const double2* __restrict__ F, const double2* __restrict__ B,
    const double2* __restrict__ C, const double* __restrict__ G,
    const int* __restrict__ off, const int* __restrict__ sz, int nline,
    const int* __restrict__ MU, const int* __restrict__ MI,
    int ne, int nu, int nu_red, int nparam, int nparam_red, int np,
    int merged,
    double2* __restrict__ A_out, double2* __restrict__ D_out)
{
    extern __shared__ double2 wsh[];
    const int lane  = threadIdx.x & 31;
    const int wid   = threadIdx.x >> 5;
    const int wpb   = blockDim.x >> 5;
    const int gwarp = blockIdx.x * wpb + wid;
    const int nwarp = gridDim.x * wpb;
    double2* Wl = wsh + (size_t)wid * nu;

    for (int e = gwarp; e < ne; e += nwarp) {
        const long long bo = (long long)e * nparam_red;
        const long long fo = (long long)e * (long long)np * nu_red;
        if constexpr (REG) {
            double2 wc[MAXUPW];
            int fu[MAXUPW];
            #pragma unroll
            for (int k = 0; k < MAXUPW; k++) {
                const int u = lane + 32*k;
                if (u < nu) {
                    fu[k] = merged ? MU[u] : u;
                    double dr = 1.0, di = 0.0;
                    for (int l = 0; l < nline; l++) {
                        const int o = off[u*nline + l], s = sz[u*nline + l];
                        if (s > 0) {
                            double fr, fi;
                            if (merged) line_den_merge<S>(B, MI, G, bo, o, s, fr, fi);
                            else line_den<S>(B, G, bo, o, s, fr, fi);
                            const double nr = dr*fr - di*fi;
                            const double ni = dr*fi + di*fr;
                            dr = nr; di = ni;
                        }
                    }
                    if (D_out) D_out[(long long)e*nu + u] = make_double2(dr, di);
                    const double den = dr*dr + di*di;
                    const double2 c = C[u];
                    wc[k] = make_double2((c.x*dr + c.y*di) / den,
                                         (c.y*dr - c.x*di) / den);
                }
            }
            for (int p = 0; p < np; p++) {
                double ar = 0.0, ai = 0.0;
                #pragma unroll
                for (int k = 0; k < MAXUPW; k++) {
                    const int u = lane + 32*k;
                    if (u < nu) {
                        const double2 f =
                            F[fo + (long long)p*nu_red + fu[k]];
                        ar += f.x*wc[k].x - f.y*wc[k].y;
                        ai += f.x*wc[k].y + f.y*wc[k].x;
                    }
                }
                const double2 A = warp_sum2(ar, ai);
                if (lane == 0) A_out[(long long)e*np + p] = A;
            }
        } else {
            for (int u = lane; u < nu; u += 32) {
                const double2 c = C[u];
                double dr = 1.0, di = 0.0;
                for (int l = 0; l < nline; l++) {
                    const int o = off[u*nline + l], s = sz[u*nline + l];
                    if (s > 0) {
                        double fr, fi;
                        if (merged) line_den_merge<S>(B, MI, G, bo, o, s, fr, fi);
                        else line_den<S>(B, G, bo, o, s, fr, fi);
                        const double nr = dr*fr - di*fi;
                        const double ni = dr*fi + di*fr;
                        dr = nr; di = ni;
                    }
                }
                if (D_out) D_out[(long long)e*nu + u] = make_double2(dr, di);
                const double den = dr*dr + di*di;
                Wl[u] = make_double2((c.x*dr + c.y*di) / den,
                                     (c.y*dr - c.x*di) / den);
            }
            __syncwarp();
            for (int p = 0; p < np; p++) {
                double ar = 0.0, ai = 0.0;
                for (int u = lane; u < nu; u += 32) {
                    const int fu = merged ? MU[u] : u;
                    const double2 f = F[fo + (long long)p*nu_red + fu];
                    const double2 wv = Wl[u];
                    ar += f.x*wv.x - f.y*wv.y;
                    ai += f.x*wv.y + f.y*wv.x;
                }
                const double2 A = warp_sum2(ar, ai);
                if (lane == 0) A_out[(long long)e*np + p] = A;
            }
        }

    }
}

template <int S>
__global__ void dense_grad_kernel_n(
    const double2* __restrict__ F, const double2* __restrict__ B,
    const double2* __restrict__ C, const double* __restrict__ G,
    const int* __restrict__ off, const int* __restrict__ sz, int nline,
    const int* __restrict__ MU, const int* __restrict__ MI,
    const double* __restrict__ dla, const double2* __restrict__ D,
    int ne, int nu, int nu_red, int nparam, int nparam_red, int np,
    int merged,
    double* __restrict__ dc_part, double* __restrict__ dg_part)
{
    const double2* DLA = reinterpret_cast<const double2*>(dla);
    for (int u = threadIdx.x; u < nu; u += blockDim.x) {
        const int fu = merged ? MU[u] : u;
        const double2 c = C[u];
        double dcr = 0.0, dci = 0.0;
        double dgr[S];

        for (int l = 0; l < nline; l++) {
            const int o = off[u*nline + l], s = sz[u*nline + l];
            const int has = s > 0;
            #pragma unroll
            for (int i = 0; i < S; i++) { dgr[i] = 0.0; }

            for (int e = blockIdx.x; e < ne; e += gridDim.x) {
                const long long bo = (long long)e * nparam_red;
                const long long fo = (long long)e * (long long)np * nu_red;
                const long long ao = (long long)e * np;
                const double2 dv = D[(long long)e*nu + u];
                const double tr = dv.x, ti = dv.y;

                double2 bv[S];
                double orr = 1.0, oi = 0.0;
                if (has) {
                    double dr, di;
                    if (merged) line_den_merge<S>(B, MI, G, bo, o, s, dr, di);
                    else line_den<S>(B, G, bo, o, s, dr, di);
                    #pragma unroll
                    for (int i = 0; i < S; i++)
                        if (i < s) bv[i] = B[bo + (merged ? MI[o+i] : o+i)];
                    const double dn = dr*dr + di*di;
                    orr = (dv.x*dr + dv.y*di) / dn;
                    oi = (dv.y*dr - dv.x*di) / dn;
                }

                double gwr = 0.0, gwi = 0.0;
                for (int p = 0; p < np; p++) {
                    const double2 da = DLA[ao+p];
                    const double2 f = F[fo + (long long)p*nu_red + fu];
                    gwr += da.x*f.x + da.y*f.y;
                    gwi += da.y*f.x - da.x*f.y;
                }
                const double den = tr*tr + ti*ti;
                const double icr = tr/den, ici = ti/den;
                if (l == 0) {
                    dcr += gwr*icr - gwi*ici;
                    dci += gwr*ici + gwi*icr;
                }
                const double gir = c.x*gwr + c.y*gwi;
                const double gii = c.x*gwi - c.y*gwr;
                const double pr = tr*tr - ti*ti, pi = -2.0*tr*ti;
                const double pn = pr*pr + pi*pi;
                const double ipr = pr/pn, ipi = -pi/pn;
                const double gdenr = -(gir*ipr - gii*ipi);
                const double gdeni = -(gir*ipi + gii*ipr);

                if (has) {
                    const double glr = gdenr*orr + gdeni*oi;
                    const double gli = gdeni*orr - gdenr*oi;
                    #pragma unroll
                    for (int i = 0; i < S; i++) {
                        if (i < s) {
                            dgr[i] += glr*bv[i].x + gli*bv[i].y;
                        }
                    }
                }
            }

            if (has) {
                #pragma unroll
                for (int i = 0; i < S; i++) {
                    if (i < s) {
                        atomicAdd(&dg_part[(long long)blockIdx.x*nparam + (o+i)],
                                  dgr[i]);
                    }
                }
            }
        }

        dc_part[(long long)blockIdx.x*2*nu + 2*u]     = dcr;
        dc_part[(long long)blockIdx.x*2*nu + 2*u + 1] = dci;
    }
}

// ---------------------------------------------------------------------------
// Real-index case: the basis is stored merged *and* as one real block per
// event, ``[Re | Im]`` split by the index maps (method: real_index).  Same
// arithmetic and layout as the general kernels above; only the F/B reads are
// replaced, so the complex basis never has to be rebuilt.
// ---------------------------------------------------------------------------

// One complex basis column of the real block: FS[e, RE[j]] + i FS[e, IM[j]].
__device__ __forceinline__ double2 ri_col(
    const double* __restrict__ FS, long long fo,
    const int* __restrict__ RE, const int* __restrict__ IM, long long j)
{
    return make_double2(FS[fo + (long long)RE[j]], FS[fo + (long long)IM[j]]);
}

// ``line_den_merge`` on a real block: the reduced column ``MI[o+i]`` is a
// pair of stored real columns, one for each half.
template <int S>
__device__ __forceinline__ void line_den_merge_ri(
    const double* __restrict__ BS, const int* __restrict__ MI,
    const int* __restrict__ BRE, const int* __restrict__ BIM,
    const double* __restrict__ G,
    long long bo, int o, int s, double& dr, double& di)
{
    dr = 0.0; di = 0.0;
    #pragma unroll
    for (int i = 0; i < S; i++) {
        if (i < s) {
            const int r = MI[o+i];
            const double g = G[o+i];
            dr += g*BS[bo + BRE[r]];
            di += g*BS[bo + BIM[r]];
        }
    }
}

template <int S, bool REG>
__global__ void dense_fwd_ri_kernel(
    const double* __restrict__ FS,   // [ne, nf], the real F block
    const double* __restrict__ BS,   // [ne, nb], the real B block
    const double2* __restrict__ C, const double* __restrict__ G,
    const int* __restrict__ off, const int* __restrict__ sz, int nline,
    const int* __restrict__ MU, const int* __restrict__ MI,
    const int* __restrict__ FRE, const int* __restrict__ FIM,
    const int* __restrict__ BRE, const int* __restrict__ BIM,
    int ne, int nu, int nu_red, int nparam, int nparam_red, int np,
    int nf, int nb,
    double2* __restrict__ A_out, double2* __restrict__ D_out)
{
    extern __shared__ double2 wsh[];
    const int lane  = threadIdx.x & 31;
    const int wid   = threadIdx.x >> 5;
    const int wpb   = blockDim.x >> 5;
    const int gwarp = blockIdx.x * wpb + wid;
    const int nwarp = gridDim.x * wpb;
    double2* Wl = wsh + (size_t)wid * nu;

    for (int e = gwarp; e < ne; e += nwarp) {
        const long long bo = (long long)e * nb;
        const long long fo = (long long)e * nf;
        if constexpr (REG) {
            double2 wc[MAXUPW];
            int fu[MAXUPW];
            #pragma unroll
            for (int k = 0; k < MAXUPW; k++) {
                const int u = lane + 32*k;
                if (u < nu) {
                    fu[k] = MU[u];
                    double dr = 1.0, di = 0.0;
                    for (int l = 0; l < nline; l++) {
                        const int o = off[u*nline + l], s = sz[u*nline + l];
                        if (s > 0) {
                            double fr, fi;
                            line_den_merge_ri<S>(BS, MI, BRE, BIM, G,
                                                   bo, o, s, fr, fi);
                            const double nr = dr*fr - di*fi;
                            const double ni = dr*fi + di*fr;
                            dr = nr; di = ni;
                        }
                    }
                    if (D_out) D_out[(long long)e*nu + u] = make_double2(dr, di);
                    const double den = dr*dr + di*di;
                    const double2 c = C[u];
                    wc[k] = make_double2((c.x*dr + c.y*di) / den,
                                         (c.y*dr - c.x*di) / den);
                }
            }
            for (int p = 0; p < np; p++) {
                double ar = 0.0, ai = 0.0;
                #pragma unroll
                for (int k = 0; k < MAXUPW; k++) {
                    const int u = lane + 32*k;
                    if (u < nu) {
                        const double2 f = ri_col(
                            FS, fo, FRE, FIM,
                            (long long)p*nu_red + fu[k]);
                        ar += f.x*wc[k].x - f.y*wc[k].y;
                        ai += f.x*wc[k].y + f.y*wc[k].x;
                    }
                }
                const double2 A = warp_sum2(ar, ai);
                if (lane == 0) A_out[(long long)e*np + p] = A;
            }
        } else {
            for (int u = lane; u < nu; u += 32) {
                const double2 c = C[u];
                double dr = 1.0, di = 0.0;
                for (int l = 0; l < nline; l++) {
                    const int o = off[u*nline + l], s = sz[u*nline + l];
                    if (s > 0) {
                        double fr, fi;
                        line_den_merge_ri<S>(BS, MI, BRE, BIM, G,
                                               bo, o, s, fr, fi);
                        const double nr = dr*fr - di*fi;
                        const double ni = dr*fi + di*fr;
                        dr = nr; di = ni;
                    }
                }
                if (D_out) D_out[(long long)e*nu + u] = make_double2(dr, di);
                const double den = dr*dr + di*di;
                Wl[u] = make_double2((c.x*dr + c.y*di) / den,
                                     (c.y*dr - c.x*di) / den);
            }
            __syncwarp();
            for (int p = 0; p < np; p++) {
                double ar = 0.0, ai = 0.0;
                for (int u = lane; u < nu; u += 32) {
                    const double2 f = ri_col(
                        FS, fo, FRE, FIM, (long long)p*nu_red + MU[u]);
                    const double2 wv = Wl[u];
                    ar += f.x*wv.x - f.y*wv.y;
                    ai += f.x*wv.y + f.y*wv.x;
                }
                const double2 A = warp_sum2(ar, ai);
                if (lane == 0) A_out[(long long)e*np + p] = A;
            }
        }

    }
}

template <int S>
__global__ void dense_grad_ri_kernel(
    const double* __restrict__ FS, const double* __restrict__ BS,
    const double2* __restrict__ C, const double* __restrict__ G,
    const int* __restrict__ off, const int* __restrict__ sz, int nline,
    const int* __restrict__ MU, const int* __restrict__ MI,
    const int* __restrict__ FRE, const int* __restrict__ FIM,
    const int* __restrict__ BRE, const int* __restrict__ BIM,
    const double* __restrict__ dla, const double2* __restrict__ D,
    int ne, int nu, int nu_red, int nparam, int nparam_red, int np,
    int nf, int nb,
    double* __restrict__ dc_part, double* __restrict__ dg_part)
{
    const double2* DLA = reinterpret_cast<const double2*>(dla);
    for (int u = threadIdx.x; u < nu; u += blockDim.x) {
        const int fu = MU[u];
        const double2 c = C[u];
        double dcr = 0.0, dci = 0.0;
        double dgr[S];

        for (int l = 0; l < nline; l++) {
            const int o = off[u*nline + l], s = sz[u*nline + l];
            const int has = s > 0;
            #pragma unroll
            for (int i = 0; i < S; i++) { dgr[i] = 0.0; }

            for (int e = blockIdx.x; e < ne; e += gridDim.x) {
                const long long bo = (long long)e * nb;
                const long long fo = (long long)e * nf;
                const long long ao = (long long)e * np;
                const double2 dv = D[(long long)e*nu + u];
                const double tr = dv.x, ti = dv.y;

                double bvr[S], bvi[S];
                double orr = 1.0, oi = 0.0;
                if (has) {
                    double dr, di;
                    line_den_merge_ri<S>(BS, MI, BRE, BIM, G,
                                           bo, o, s, dr, di);
                    #pragma unroll
                    for (int i = 0; i < S; i++)
                        if (i < s) {
                            const int r = MI[o+i];
                            bvr[i] = BS[bo + BRE[r]];
                            bvi[i] = BS[bo + BIM[r]];
                        }
                    const double dn = dr*dr + di*di;
                    orr = (dv.x*dr + dv.y*di) / dn;
                    oi = (dv.y*dr - dv.x*di) / dn;
                }

                double gwr = 0.0, gwi = 0.0;
                for (int p = 0; p < np; p++) {
                    const double2 da = DLA[ao+p];
                    const double2 f = ri_col(
                        FS, fo, FRE, FIM, (long long)p*nu_red + fu);
                    gwr += da.x*f.x + da.y*f.y;
                    gwi += da.y*f.x - da.x*f.y;
                }
                const double den = tr*tr + ti*ti;
                const double icr = tr/den, ici = ti/den;
                if (l == 0) {
                    dcr += gwr*icr - gwi*ici;
                    dci += gwr*ici + gwi*icr;
                }
                const double gir = c.x*gwr + c.y*gwi;
                const double gii = c.x*gwi - c.y*gwr;
                const double pr = tr*tr - ti*ti, pi = -2.0*tr*ti;
                const double pn = pr*pr + pi*pi;
                const double ipr = pr/pn, ipi = -pi/pn;
                const double gdenr = -(gir*ipr - gii*ipi);
                const double gdeni = -(gir*ipi + gii*ipr);

                if (has) {
                    const double glr = gdenr*orr + gdeni*oi;
                    const double gli = gdeni*orr - gdenr*oi;
                    #pragma unroll
                    for (int i = 0; i < S; i++) {
                        if (i < s) {
                            dgr[i] += glr*bvr[i] + gli*bvi[i];
                        }
                    }
                }
            }

            if (has) {
                #pragma unroll
                for (int i = 0; i < S; i++) {
                    if (i < s) {
                        atomicAdd(&dg_part[(long long)blockIdx.x*nparam + (o+i)],
                                  dgr[i]);
                    }
                }
            }
        }

        dc_part[(long long)blockIdx.x*2*nu + 2*u]     = dcr;
        dc_part[(long long)blockIdx.x*2*nu + 2*u + 1] = dci;
    }
}

// ---------------------------------------------------------------------------
// Basis-index case: the stored real block holds only the *basis* columns of
// method: basis_index, and every reduced column is a sparse transform of
// them with a sparse CSR transform.  Same arithmetic and layout
// as the general kernels; only the F/B reads are replaced, so neither the
// complex basis nor the dropped columns are rebuilt.
// ---------------------------------------------------------------------------

// One reduced column of a stored real block: exact CSR row ``j``, whose
// entries are ``VAL[i] * basis[COL[i]]`` for ``PTR[j] <= i < PTR[j+1]``.  The
// basis column ``p`` is ``STORE[so + RE[p]] + i STORE[so + IM[p]]``.
template <int T>
__device__ __forceinline__ double2 bi_col(
    const double* __restrict__ STORE, long long so,
    const int* __restrict__ RE, const int* __restrict__ IM,
    const int* __restrict__ PTR, const int* __restrict__ COL,
    const double2* __restrict__ VAL, long long j)
{
    double2 out = make_double2(0.0, 0.0);
    // T is the widest row: unroll the bound so short rows leave the loop
    // early instead of testing a sentinel weight
    #pragma unroll
    for (int t = 0; t < T; t++) {
        const int i = PTR[j] + t;
        if (i >= PTR[j+1]) break;
        const int p = COL[i];
        const double2 w = VAL[i];
        const double2 b =
            make_double2(STORE[so + (long long)RE[p]],
                         STORE[so + (long long)IM[p]]);
        out.x += w.x*b.x - w.y*b.y;
        out.y += w.x*b.y + w.y*b.x;
    }
    return out;
}

// ``line_den_merge`` on a basis-index real block: the reduced column
// ``MI[o+i]`` is the sparse CSR transform ``BI_COL`` of the stored basis.
template <int S, int T>
__device__ __forceinline__ void line_den_merge_bi(
    const double* __restrict__ BS, const int* __restrict__ MI,
    const int* __restrict__ BRE, const int* __restrict__ BIM,
    const int* __restrict__ PTR, const int* __restrict__ COL,
    const double2* __restrict__ VAL,
    const double* __restrict__ G,
    long long bo, int o, int s, double& dr, double& di)
{
    dr = 0.0; di = 0.0;
    #pragma unroll
    for (int i = 0; i < S; i++) {
        if (i < s) {
            const int r = MI[o+i];
            const double g = G[o+i];
            const double2 b = bi_col<T>(BS, bo, BRE, BIM, PTR, COL, VAL, r);
            dr += g*b.x;
            di += g*b.y;
        }
    }
}

template <int S, bool REG, int T>
__global__ void dense_fwd_bi_kernel(
    const double* __restrict__ FS,   // [ne, nf], the real F block
    const double* __restrict__ BS,   // [ne, nb], the real B block
    const double2* __restrict__ C, const double* __restrict__ G,
    const int* __restrict__ off, const int* __restrict__ sz, int nline,
    const int* __restrict__ MU, const int* __restrict__ MI,
    const int* __restrict__ FRE, const int* __restrict__ FIM,
    const int* __restrict__ BRE, const int* __restrict__ BIM,
    const int* __restrict__ PF_PTR, const int* __restrict__ PF_COL,
    const double2* __restrict__ PF_VAL,
    const int* __restrict__ PB_PTR, const int* __restrict__ PB_COL,
    const double2* __restrict__ PB_VAL,
    int ne, int nu, int nu_red, int nparam, int nparam_red, int np,
    int nf, int nb,
    double2* __restrict__ A_out, double2* __restrict__ D_out)
{
    extern __shared__ double2 wsh[];
    const int lane  = threadIdx.x & 31;
    const int wid   = threadIdx.x >> 5;
    const int wpb   = blockDim.x >> 5;
    const int gwarp = blockIdx.x * wpb + wid;
    const int nwarp = gridDim.x * wpb;
    double2* Wl = wsh + (size_t)wid * nu;

    for (int e = gwarp; e < ne; e += nwarp) {
        const long long bo = (long long)e * nb;
        const long long fo = (long long)e * nf;
        if constexpr (REG) {
            double2 wc[MAXUPW];
            int fu[MAXUPW];
            #pragma unroll
            for (int k = 0; k < MAXUPW; k++) {
                const int u = lane + 32*k;
                if (u < nu) {
                    fu[k] = MU[u];
                    double dr = 1.0, di = 0.0;
                    for (int l = 0; l < nline; l++) {
                        const int o = off[u*nline + l], s = sz[u*nline + l];
                        if (s > 0) {
                            double fr, fi;
                            line_den_merge_bi<S, T>(BS, MI, BRE, BIM,
                                                 PB_PTR, PB_COL, PB_VAL,
                                                 G, bo, o, s, fr, fi);
                            const double nr = dr*fr - di*fi;
                            const double ni = dr*fi + di*fr;
                            dr = nr; di = ni;
                        }
                    }
                    if (D_out) D_out[(long long)e*nu + u] = make_double2(dr, di);
                    const double den = dr*dr + di*di;
                    const double2 c = C[u];
                    wc[k] = make_double2((c.x*dr + c.y*di) / den,
                                         (c.y*dr - c.x*di) / den);
                }
            }
            for (int p = 0; p < np; p++) {
                double ar = 0.0, ai = 0.0;
                #pragma unroll
                for (int k = 0; k < MAXUPW; k++) {
                    const int u = lane + 32*k;
                    if (u < nu) {
                        const double2 f = bi_col<T>(
                            FS, fo, FRE, FIM, PF_PTR, PF_COL, PF_VAL,
                            (long long)p*nu_red + fu[k]);
                        ar += f.x*wc[k].x - f.y*wc[k].y;
                        ai += f.x*wc[k].y + f.y*wc[k].x;
                    }
                }
                const double2 A = warp_sum2(ar, ai);
                if (lane == 0) A_out[(long long)e*np + p] = A;
            }
        } else {
            for (int u = lane; u < nu; u += 32) {
                const double2 c = C[u];
                double dr = 1.0, di = 0.0;
                for (int l = 0; l < nline; l++) {
                    const int o = off[u*nline + l], s = sz[u*nline + l];
                    if (s > 0) {
                        double fr, fi;
                        line_den_merge_bi<S, T>(BS, MI, BRE, BIM,
                                             PB_PTR, PB_COL, PB_VAL,
                                             G, bo, o, s, fr, fi);
                        const double nr = dr*fr - di*fi;
                        const double ni = dr*fi + di*fr;
                        dr = nr; di = ni;
                    }
                }
                if (D_out) D_out[(long long)e*nu + u] = make_double2(dr, di);
                const double den = dr*dr + di*di;
                Wl[u] = make_double2((c.x*dr + c.y*di) / den,
                                     (c.y*dr - c.x*di) / den);
            }
            __syncwarp();
            for (int p = 0; p < np; p++) {
                double ar = 0.0, ai = 0.0;
                for (int u = lane; u < nu; u += 32) {
                    const double2 f = bi_col<T>(
                        FS, fo, FRE, FIM, PF_PTR, PF_COL, PF_VAL,
                        (long long)p*nu_red + MU[u]);
                    const double2 wv = Wl[u];
                    ar += f.x*wv.x - f.y*wv.y;
                    ai += f.x*wv.y + f.y*wv.x;
                }
                const double2 A = warp_sum2(ar, ai);
                if (lane == 0) A_out[(long long)e*np + p] = A;
            }
        }

    }
}

template <int S, int T>
__global__ void dense_grad_bi_kernel(
    const double* __restrict__ FS, const double* __restrict__ BS,
    const double2* __restrict__ C, const double* __restrict__ G,
    const int* __restrict__ off, const int* __restrict__ sz, int nline,
    const int* __restrict__ MU, const int* __restrict__ MI,
    const int* __restrict__ FRE, const int* __restrict__ FIM,
    const int* __restrict__ BRE, const int* __restrict__ BIM,
    const int* __restrict__ PF_PTR, const int* __restrict__ PF_COL,
    const double2* __restrict__ PF_VAL,
    const int* __restrict__ PB_PTR, const int* __restrict__ PB_COL,
    const double2* __restrict__ PB_VAL,
    const double* __restrict__ dla, const double2* __restrict__ D,
    int ne, int nu, int nu_red, int nparam, int nparam_red, int np,
    int nf, int nb,
    double* __restrict__ dc_part, double* __restrict__ dg_part)
{
    const double2* DLA = reinterpret_cast<const double2*>(dla);
    for (int u = threadIdx.x; u < nu; u += blockDim.x) {
        const int fu = MU[u];
        const double2 c = C[u];
        double dcr = 0.0, dci = 0.0;
        double dgr[S];

        for (int l = 0; l < nline; l++) {
            const int o = off[u*nline + l], s = sz[u*nline + l];
            const int has = s > 0;
            #pragma unroll
            for (int i = 0; i < S; i++) { dgr[i] = 0.0; }

            for (int e = blockIdx.x; e < ne; e += gridDim.x) {
                const long long bo = (long long)e * nb;
                const long long fo = (long long)e * nf;
                const long long ao = (long long)e * np;
                const double2 dv = D[(long long)e*nu + u];
                const double tr = dv.x, ti = dv.y;

                double bvr[S], bvi[S];
                double orr = 1.0, oi = 0.0;
                if (has) {
                    double dr, di;
                    line_den_merge_bi<S, T>(BS, MI, BRE, BIM,
                                         PB_PTR, PB_COL, PB_VAL,
                                         G, bo, o, s, dr, di);
                    #pragma unroll
                    for (int i = 0; i < S; i++)
                        if (i < s) {
                            const double2 b = bi_col<T>(
                                BS, bo, BRE, BIM, PB_PTR, PB_COL, PB_VAL,
                                MI[o+i]);
                            bvr[i] = b.x;
                            bvi[i] = b.y;
                        }
                    const double dn = dr*dr + di*di;
                    orr = (dv.x*dr + dv.y*di) / dn;
                    oi = (dv.y*dr - dv.x*di) / dn;
                }

                double gwr = 0.0, gwi = 0.0;
                for (int p = 0; p < np; p++) {
                    const double2 da = DLA[ao+p];
                    const double2 f = bi_col<T>(
                        FS, fo, FRE, FIM, PF_PTR, PF_COL, PF_VAL,
                        (long long)p*nu_red + fu);
                    gwr += da.x*f.x + da.y*f.y;
                    gwi += da.y*f.x - da.x*f.y;
                }
                const double den = tr*tr + ti*ti;
                const double icr = tr/den, ici = ti/den;
                if (l == 0) {
                    dcr += gwr*icr - gwi*ici;
                    dci += gwr*ici + gwi*icr;
                }
                const double gir = c.x*gwr + c.y*gwi;
                const double gii = c.x*gwi - c.y*gwr;
                const double pr = tr*tr - ti*ti, pi = -2.0*tr*ti;
                const double pn = pr*pr + pi*pi;
                const double ipr = pr/pn, ipi = -pi/pn;
                const double gdenr = -(gir*ipr - gii*ipi);
                const double gdeni = -(gir*ipi + gii*ipr);

                if (has) {
                    const double glr = gdenr*orr + gdeni*oi;
                    const double gli = gdeni*orr - gdenr*oi;
                    #pragma unroll
                    for (int i = 0; i < S; i++) {
                        if (i < s) {
                            dgr[i] += glr*bvr[i] + gli*bvi[i];
                        }
                    }
                }
            }

            if (has) {
                #pragma unroll
                for (int i = 0; i < S; i++) {
                    if (i < s) {
                        atomicAdd(&dg_part[(long long)blockIdx.x*nparam + (o+i)],
                                  dgr[i]);
                    }
                }
            }
        }

        dc_part[(long long)blockIdx.x*2*nu + 2*u]     = dcr;
        dc_part[(long long)blockIdx.x*2*nu + 2*u + 1] = dci;
    }
}

struct Args {
    // ``dla`` is dL/dA (complex) in gradient mode and null in forward mode.
    const double *fr, *btr, *cr, *gtr; const int *off, *sz;
    const int *mur, *mir; const double *dla;
    int ne, nu, nu_red, nparam, nparam_red, np, nline, nblk, merged;
    double *a_out, *d_out, *dc_part, *dg_part, *dc_out, *dg_out;
    cudaStream_t s;
};

static int pick_wpb(int nu) {
    int wpb = NWARP;
    while (wpb > 1 && (size_t)wpb * nu * sizeof(double2) > SHMEM_LIMIT) wpb--;
    return wpb;
}

template <int S, bool REG>
static void run(const Args& a, int mode) {
    const double2* F = reinterpret_cast<const double2*>(a.fr);
    const double2* B = reinterpret_cast<const double2*>(a.btr);
    const double2* C = reinterpret_cast<const double2*>(a.cr);
    const double* G = a.gtr;
    double2* A = reinterpret_cast<double2*>(a.a_out);
    int wpb = REG ? NWARP : pick_wpb(a.nu);
    size_t shmem = REG ? 0 : (size_t)wpb * a.nu * sizeof(double2);
    int nb_f = a.ne < 4096 ? (a.ne + wpb - 1) / wpb : 2048;
    if (nb_f < 1) nb_f = 1;
    if (mode != 2)
        dense_fwd_kernel<S, REG><<<nb_f, wpb*32, shmem, a.s>>>(
            F, B, C, G, a.off, a.sz, a.mur, a.mir,
            a.ne, a.nu, a.nu_red, a.nparam, a.nparam_red, a.np, a.merged, A);
    if (mode != 1) {
        cudaMemsetAsync(a.dg_part, 0,
                        (size_t)a.nblk * a.nparam * sizeof(double), a.s);
        int gb = a.nu <= 128 ? 128 : 256;
        dense_grad_kernel<S><<<a.nblk, gb, 0, a.s>>>(
            F, B, C, G, a.off, a.sz, a.mur, a.mir, a.dla,
            a.ne, a.nu, a.nu_red, a.nparam, a.nparam_red, a.np, a.merged,
            a.dc_part, a.dg_part);
        reduce_k<<<(2*a.nu+255)/256, 256, 0, a.s>>>(
            a.dc_part, a.nblk, 2*a.nu, 0, 2*a.nu, a.dc_out);
        reduce_k<<<(a.nparam+255)/256, 256, 0, a.s>>>(
            a.dg_part, a.nblk, a.nparam, 0, a.nparam, a.dg_out);
    }
}

template <int S, bool REG>
static void run_n(const Args& a, int mode) {
    const double2* F = reinterpret_cast<const double2*>(a.fr);
    const double2* B = reinterpret_cast<const double2*>(a.btr);
    const double2* C = reinterpret_cast<const double2*>(a.cr);
    const double* G = a.gtr;
    double2* A = reinterpret_cast<double2*>(a.a_out);
    double2* D = reinterpret_cast<double2*>(a.d_out);
    int wpb = REG ? NWARP : pick_wpb(a.nu);
    size_t shmem = REG ? 0 : (size_t)wpb * a.nu * sizeof(double2);
    int nb_f = a.ne < 4096 ? (a.ne + wpb - 1) / wpb : 2048;
    if (nb_f < 1) nb_f = 1;
    if (mode != 2)
        dense_fwd_kernel_n<S, REG><<<nb_f, wpb*32, shmem, a.s>>>(
            F, B, C, G, a.off, a.sz, a.nline, a.mur, a.mir,
            a.ne, a.nu, a.nu_red, a.nparam, a.nparam_red, a.np, a.merged,
            A, D);
    if (mode != 1) {
        cudaMemsetAsync(a.dg_part, 0,
                        (size_t)a.nblk * a.nparam * sizeof(double), a.s);
        int gb = a.nu <= 128 ? 128 : 256;
        dense_grad_kernel_n<S><<<a.nblk, gb, 0, a.s>>>(
            F, B, C, G, a.off, a.sz, a.nline, a.mur, a.mir, a.dla, D,
            a.ne, a.nu, a.nu_red, a.nparam, a.nparam_red, a.np, a.merged,
            a.dc_part, a.dg_part);
        reduce_k<<<(2*a.nu+255)/256, 256, 0, a.s>>>(
            a.dc_part, a.nblk, 2*a.nu, 0, 2*a.nu, a.dc_out);
        reduce_k<<<(a.nparam+255)/256, 256, 0, a.s>>>(
            a.dg_part, a.nblk, a.nparam, 0, a.nparam, a.dg_out);
    }
}

template <bool REG>
static void dispatch_s(const Args& a, int mode, int maxs, bool general) {
    switch (maxs) {
        case 1: general ? run_n<1, REG>(a, mode) : run<1, REG>(a, mode); break;
        case 2: general ? run_n<2, REG>(a, mode) : run<2, REG>(a, mode); break;
        case 3: general ? run_n<3, REG>(a, mode) : run<3, REG>(a, mode); break;
        case 4: general ? run_n<4, REG>(a, mode) : run<4, REG>(a, mode); break;
        case 5: general ? run_n<5, REG>(a, mode) : run<5, REG>(a, mode); break;
        case 6: general ? run_n<6, REG>(a, mode) : run<6, REG>(a, mode); break;
        case 7: general ? run_n<7, REG>(a, mode) : run<7, REG>(a, mode); break;
        case 8: general ? run_n<8, REG>(a, mode) : run<8, REG>(a, mode); break;
        default: general ? run_n<16, REG>(a, mode) : run<16, REG>(a, mode); break;
    }
}

static void dispatch(const Args& a, int mode, int maxs, bool general) {
    // The gradient does not use the flag; the forward keeps W in registers
    // whenever every chain gets a lane slot.
    if (a.nu <= 32 * MAXUPW)
        dispatch_s<true>(a, mode, maxs, general);
    else
        dispatch_s<false>(a, mode, maxs, general);
}

extern "C" void launch_dense_factor(
    const double* fr, const double* btr, const double* cr, const double* gtr,
    const int* off, const int* sz, const int* mur, const int* mir,
    const double* dla,
    int ne, int nu, int nu_red, int nparam, int nparam_red, int np, int nline,
    int nblk, int mode, int maxs, int merged,
    double* a_out, double* d_out,
    double* dc_part, double* dg_part,
    double* dc_out, double* dg_out, cudaStream_t s)
{
    Args a{fr, btr, cr, gtr, off, sz, mur, mir, dla,
           ne, nu, nu_red, nparam, nparam_red, np, nline, nblk, merged,
           a_out, d_out, dc_part, dg_part, dc_out, dg_out, s};
    dispatch(a, mode, maxs, false);
}

extern "C" void launch_dense_factor_n(
    const double* fr, const double* btr, const double* cr, const double* gtr,
    const int* off, const int* sz, const int* mur, const int* mir,
    const double* dla,
    int ne, int nu, int nu_red, int nparam, int nparam_red, int np, int nline,
    int nblk, int mode, int maxs, int merged,
    double* a_out, double* d_out,
    double* dc_part, double* dg_part,
    double* dc_out, double* dg_out, cudaStream_t s)
{
    Args a{fr, btr, cr, gtr, off, sz, mur, mir, dla,
           ne, nu, nu_red, nparam, nparam_red, np, nline, nblk, merged,
           a_out, d_out, dc_part, dg_part, dc_out, dg_out, s};
    dispatch(a, mode, maxs, true);
}

// ---------------------------------------------------------------------------
// Real-index entry point: the same general launch, with the split F/B blocks
// and their read maps in place of the complex ones.
// ---------------------------------------------------------------------------

struct RiArgs {
    const double *fs, *bs, *cr, *gtr; const int *off, *sz;
    const int *mur, *mir, *fre, *fim, *bre, *bim; const double *dla;
    int ne, nu, nu_red, nparam, nparam_red, np, nf, nb, nline, nblk;
    double *a_out, *d_out, *dc_part, *dg_part, *dc_out, *dg_out;
    cudaStream_t s;
};

template <int S, bool REG>
static void run_ri(const RiArgs& a, int mode) {
    const double2* C = reinterpret_cast<const double2*>(a.cr);
    const double* G = a.gtr;
    double2* A = reinterpret_cast<double2*>(a.a_out);
    double2* D = reinterpret_cast<double2*>(a.d_out);
    int wpb = REG ? NWARP : pick_wpb(a.nu);
    size_t shmem = REG ? 0 : (size_t)wpb * a.nu * sizeof(double2);
    int nb_f = a.ne < 4096 ? (a.ne + wpb - 1) / wpb : 2048;
    if (nb_f < 1) nb_f = 1;
    if (mode != 2)
        dense_fwd_ri_kernel<S, REG><<<nb_f, wpb*32, shmem, a.s>>>(
            a.fs, a.bs, C, G, a.off, a.sz, a.nline, a.mur, a.mir,
            a.fre, a.fim, a.bre, a.bim,
            a.ne, a.nu, a.nu_red, a.nparam, a.nparam_red, a.np, a.nf, a.nb,
            A, D);
    if (mode != 1) {
        cudaMemsetAsync(a.dg_part, 0,
                        (size_t)a.nblk * a.nparam * sizeof(double), a.s);
        int gb = a.nu <= 128 ? 128 : 256;
        dense_grad_ri_kernel<S><<<a.nblk, gb, 0, a.s>>>(
            a.fs, a.bs, C, G, a.off, a.sz, a.nline, a.mur, a.mir,
            a.fre, a.fim, a.bre, a.bim, a.dla, D,
            a.ne, a.nu, a.nu_red, a.nparam, a.nparam_red, a.np, a.nf, a.nb,
            a.dc_part, a.dg_part);
        reduce_k<<<(2*a.nu+255)/256, 256, 0, a.s>>>(
            a.dc_part, a.nblk, 2*a.nu, 0, 2*a.nu, a.dc_out);
        reduce_k<<<(a.nparam+255)/256, 256, 0, a.s>>>(
            a.dg_part, a.nblk, a.nparam, 0, a.nparam, a.dg_out);
    }
}

template <bool REG>
static void dispatch_ri_s(const RiArgs& a, int mode, int maxs) {
    switch (maxs) {
        case 1: run_ri<1, REG>(a, mode); break;
        case 2: run_ri<2, REG>(a, mode); break;
        case 3: run_ri<3, REG>(a, mode); break;
        case 4: run_ri<4, REG>(a, mode); break;
        case 5: run_ri<5, REG>(a, mode); break;
        case 6: run_ri<6, REG>(a, mode); break;
        case 7: run_ri<7, REG>(a, mode); break;
        case 8: run_ri<8, REG>(a, mode); break;
        default: run_ri<16, REG>(a, mode); break;
    }
}

static void dispatch_ri(const RiArgs& a, int mode, int maxs) {
    if (a.nu <= 32 * MAXUPW)
        dispatch_ri_s<true>(a, mode, maxs);
    else
        dispatch_ri_s<false>(a, mode, maxs);
}

extern "C" void launch_dense_factor_ri(
    const double* fs, const double* bs, const double* cr, const double* gtr,
    const int* off, const int* sz, const int* mur, const int* mir,
    const int* fre, const int* fim, const int* bre, const int* bim,
    const double* dla,
    int ne, int nu, int nu_red, int nparam, int nparam_red, int np,
    int nf, int nb, int nline,
    int nblk, int mode, int maxs,
    double* a_out, double* d_out,
    double* dc_part, double* dg_part,
    double* dc_out, double* dg_out, cudaStream_t s)
{
    RiArgs a{fs, bs, cr, gtr, off, sz, mur, mir, fre, fim, bre, bim, dla,
               ne, nu, nu_red, nparam, nparam_red, np, nf, nb, nline, nblk,
               a_out, d_out, dc_part, dg_part, dc_out, dg_out, s};
    dispatch_ri(a, mode, maxs);
}

// ---------------------------------------------------------------------------
// Basis-index entry point: the same general launch, with the stored basis
// columns and their sparse transform in place of the complex ones.
// ---------------------------------------------------------------------------

struct BiArgs {
    const double *fs, *bs, *cr, *gtr; const int *off, *sz;
    const int *mur, *mir, *fre, *fim, *bre, *bim;
    const int *f_ptr, *f_col, *b_ptr, *b_col;
    const double2 *f_val, *b_val;
    const double *dla;
    int ne, nu, nu_red, nparam, nparam_red, np, nf, nb;
    int terms, nline, nblk;
    double *a_out, *d_out, *dc_part, *dg_part, *dc_out, *dg_out;
    cudaStream_t s;
};

template <int S, bool REG, int T>
static void run_bi(const BiArgs& a, int mode) {
    const double2* C = reinterpret_cast<const double2*>(a.cr);
    const double* G = a.gtr;
    double2* A = reinterpret_cast<double2*>(a.a_out);
    double2* D = reinterpret_cast<double2*>(a.d_out);
    int wpb = REG ? NWARP : pick_wpb(a.nu);
    size_t shmem = REG ? 0 : (size_t)wpb * a.nu * sizeof(double2);
    int nb_f = a.ne < 4096 ? (a.ne + wpb - 1) / wpb : 2048;
    if (nb_f < 1) nb_f = 1;
    if (mode != 2)
        dense_fwd_bi_kernel<S, REG, T><<<nb_f, wpb*32, shmem, a.s>>>(
            a.fs, a.bs, C, G, a.off, a.sz, a.nline, a.mur, a.mir,
            a.fre, a.fim, a.bre, a.bim,
            a.f_ptr, a.f_col, a.f_val, a.b_ptr, a.b_col, a.b_val,
            a.ne, a.nu, a.nu_red, a.nparam, a.nparam_red, a.np, a.nf, a.nb,
            A, D);
    if (mode != 1) {
        cudaMemsetAsync(a.dg_part, 0,
                        (size_t)a.nblk * a.nparam * sizeof(double), a.s);
        int gb = a.nu <= 128 ? 128 : 256;
        dense_grad_bi_kernel<S, T><<<a.nblk, gb, 0, a.s>>>(
            a.fs, a.bs, C, G, a.off, a.sz, a.nline, a.mur, a.mir,
            a.fre, a.fim, a.bre, a.bim,
            a.f_ptr, a.f_col, a.f_val, a.b_ptr, a.b_col, a.b_val,
            a.dla, D, a.ne, a.nu, a.nu_red, a.nparam, a.nparam_red, a.np,
            a.nf, a.nb, a.dc_part, a.dg_part);
        reduce_k<<<(2*a.nu+255)/256, 256, 0, a.s>>>(
            a.dc_part, a.nblk, 2*a.nu, 0, 2*a.nu, a.dc_out);
        reduce_k<<<(a.nparam+255)/256, 256, 0, a.s>>>(
            a.dg_part, a.nblk, a.nparam, 0, a.nparam, a.dg_out);
    }
}

template <bool REG, int T>
static void dispatch_bi_t(const BiArgs& a, int mode, int maxs) {
    switch (maxs) {
        case 1: run_bi<1, REG, T>(a, mode); break;
        case 2: run_bi<2, REG, T>(a, mode); break;
        case 3: run_bi<3, REG, T>(a, mode); break;
        case 4: run_bi<4, REG, T>(a, mode); break;
        case 5: run_bi<5, REG, T>(a, mode); break;
        case 6: run_bi<6, REG, T>(a, mode); break;
        case 7: run_bi<7, REG, T>(a, mode); break;
        case 8: run_bi<8, REG, T>(a, mode); break;
        default: run_bi<16, REG, T>(a, mode); break;
    }
}

template <bool REG>
static void dispatch_bi_s(const BiArgs& a, int mode, int maxs) {
    switch (a.terms) {
        case 1: dispatch_bi_t<REG, 1>(a, mode, maxs); break;
        case 2: dispatch_bi_t<REG, 2>(a, mode, maxs); break;
        case 3: dispatch_bi_t<REG, 3>(a, mode, maxs); break;
        default: dispatch_bi_t<REG, 4>(a, mode, maxs); break;
    }
}

static void dispatch_bi(const BiArgs& a, int mode, int maxs) {
    if (a.nu <= 32 * MAXUPW)
        dispatch_bi_s<true>(a, mode, maxs);
    else
        dispatch_bi_s<false>(a, mode, maxs);
}


extern "C" void launch_dense_factor_bi(
    const double* fs, const double* bs, const double* cr, const double* gtr,
    const int* off, const int* sz, const int* mur, const int* mir,
    const int* fre, const int* fim, const int* bre, const int* bim,
    const int* f_ptr, const int* f_col, const double* f_val,
    const int* b_ptr, const int* b_col, const double* b_val,
    const double* dla,
    int ne, int nu, int nu_red, int nparam, int nparam_red, int np,
    int nf, int nb, int terms,
    int nline, int nblk, int mode, int maxs,
    double* a_out, double* d_out,
    double* dc_part, double* dg_part,
    double* dc_out, double* dg_out, cudaStream_t s)
{
    BiArgs a{fs, bs, cr, gtr, off, sz, mur, mir, fre, fim, bre, bim,
             f_ptr, f_col, b_ptr, b_col,
             reinterpret_cast<const double2*>(f_val),
             reinterpret_cast<const double2*>(b_val),
             dla, ne, nu, nu_red, nparam, nparam_red, np, nf, nb,
             terms, nline, nblk,
             a_out, d_out, dc_part, dg_part, dc_out, dg_out, s};
    dispatch_bi(a, mode, maxs);
}
