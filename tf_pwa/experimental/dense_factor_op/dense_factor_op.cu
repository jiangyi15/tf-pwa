#include <cuda_runtime.h>

#define NWARP 8
#define SHMEM_LIMIT 49152   /* default dynamic shared memory per block */
#define MAXUPW 4            /* chains per lane kept in registers (nu <= 128) */

// den[u,e] = prod_l sum_i g[off[u*nline+l]+i] B[e, off[u*nline+l]+i]
// A_p[e]   = sum_u F[p,e,u] * C[u] / den[u,e]
// P[e]     = sum_p |A_p[e]|^2
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
    const double* __restrict__ w, int ne, int nu, int nparam, int np,
    double2* __restrict__ A_out,     // [ne * np]
    double* __restrict__ p_out,
    double* __restrict__ q_out)
{
    extern __shared__ double2 wsh[];
    const int lane  = threadIdx.x & 31;
    const int wid   = threadIdx.x >> 5;
    const int wpb   = blockDim.x >> 5;
    const int gwarp = blockIdx.x * wpb + wid;
    const int nwarp = gridDim.x * wpb;
    double2* Wl = wsh + (size_t)wid * nu;
    double qacc = 0.0;

    for (int e = gwarp; e < ne; e += nwarp) {
        const long long bo = (long long)e * nparam;
        const long long fo = (long long)e * (long long)np * nu;
        double Psum = 0.0;

        if constexpr (REG) {
            double2 wc[MAXUPW];
            #pragma unroll
            for (int k = 0; k < MAXUPW; k++) {
                const int u = lane + 32*k;
                if (u < nu) {
                    const int o = off[u], s = sz[u];
                    double dr, di;
                    line_den<S>(B, G, bo, o, s, dr, di);
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
                        const double2 f = F[fo + (long long)p*nu + u];
                        ar += f.x*wc[k].x - f.y*wc[k].y;
                        ai += f.x*wc[k].y + f.y*wc[k].x;
                    }
                }
                const double2 A = warp_sum2(ar, ai);
                if (lane == 0) {
                    A_out[(long long)e*np + p] = A;
                    Psum += A.x*A.x + A.y*A.y;
                }
            }
        } else {
            for (int u = lane; u < nu; u += 32) {
                const int o = off[u], s = sz[u];
                double dr, di;
                line_den<S>(B, G, bo, o, s, dr, di);
                const double den = dr*dr + di*di;
                const double2 c = C[u];
                Wl[u] = make_double2((c.x*dr + c.y*di) / den,
                                     (c.y*dr - c.x*di) / den);
            }
            __syncwarp();
            for (int p = 0; p < np; p++) {
                double ar = 0.0, ai = 0.0;
                for (int u = lane; u < nu; u += 32) {
                    const double2 f = F[fo + (long long)p*nu + u];
                    const double2 wv = Wl[u];
                    ar += f.x*wv.x - f.y*wv.y;
                    ai += f.x*wv.y + f.y*wv.x;
                }
                const double2 A = warp_sum2(ar, ai);
                if (lane == 0) {
                    A_out[(long long)e*np + p] = A;
                    Psum += A.x*A.x + A.y*A.y;
                }
            }
        }

        if (lane == 0) {
            if (q_out) qacc += w[e]*Psum;
            p_out[e] = Psum;
        }
    }
    if (lane == 0 && q_out) atomicAdd(q_out, qacc);
}

template <int S>
__global__ void dense_grad_kernel(
    const double2* __restrict__ F,
    const double2* __restrict__ B,
    const double2* __restrict__ C,
    const double* __restrict__ G,
    const int* __restrict__ off, const int* __restrict__ sz,
    const double* __restrict__ w,
    const double2* __restrict__ A,   // [ne * np]
    int ne, int nu, int nparam, int np,
    double* __restrict__ dc_part, double* __restrict__ dg_part)
{
    for (int u = threadIdx.x; u < nu; u += blockDim.x) {
        const int o = off[u], s = sz[u];
        const double2 c = C[u];
        double dcr = 0.0, dci = 0.0;
        double dgr[S];
        #pragma unroll
        for (int i = 0; i < S; i++) { dgr[i] = 0.0; }

        for (int e = blockIdx.x; e < ne; e += gridDim.x) {
            const double we = w[e];
            const long long bo = (long long)e * nparam;
            const long long fo = (long long)e * (long long)np * nu;
            const long long ao = (long long)e * np;
            double2 bv[S];
            #pragma unroll
            for (int i = 0; i < S; i++)
                if (i < s) bv[i] = B[bo+o+i];
            double dr, di;
            line_den<S>(B, G, bo, o, s, dr, di);
            double gwr = 0.0, gwi = 0.0;
            for (int p = 0; p < np; p++) {
                const double2 a = A[ao+p];
                const double2 f = F[fo + (long long)p*nu + u];
                gwr += 2.0*we*( f.x*a.x + f.y*a.y);
                gwi += 2.0*we*( f.x*a.y - f.y*a.x);
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
    const double* __restrict__ w, int ne, int nu, int nparam, int np,
    double2* __restrict__ A_out, double* __restrict__ p_out,
    double* __restrict__ q_out, double2* __restrict__ D_out)
{
    extern __shared__ double2 wsh[];
    const int lane  = threadIdx.x & 31;
    const int wid   = threadIdx.x >> 5;
    const int wpb   = blockDim.x >> 5;
    const int gwarp = blockIdx.x * wpb + wid;
    const int nwarp = gridDim.x * wpb;
    double2* Wl = wsh + (size_t)wid * nu;
    double qacc = 0.0;

    for (int e = gwarp; e < ne; e += nwarp) {
        const long long bo = (long long)e * nparam;
        const long long fo = (long long)e * (long long)np * nu;
        double Psum = 0.0;

        if constexpr (REG) {
            double2 wc[MAXUPW];
            #pragma unroll
            for (int k = 0; k < MAXUPW; k++) {
                const int u = lane + 32*k;
                if (u < nu) {
                    double dr = 1.0, di = 0.0;
                    for (int l = 0; l < nline; l++) {
                        const int o = off[u*nline + l], s = sz[u*nline + l];
                        if (s > 0) {
                            double fr, fi;
                            line_den<S>(B, G, bo, o, s, fr, fi);
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
                        const double2 f = F[fo + (long long)p*nu + u];
                        ar += f.x*wc[k].x - f.y*wc[k].y;
                        ai += f.x*wc[k].y + f.y*wc[k].x;
                    }
                }
                const double2 A = warp_sum2(ar, ai);
                if (lane == 0) {
                    A_out[(long long)e*np + p] = A;
                    Psum += A.x*A.x + A.y*A.y;
                }
            }
        } else {
            for (int u = lane; u < nu; u += 32) {
                const double2 c = C[u];
                double dr = 1.0, di = 0.0;
                for (int l = 0; l < nline; l++) {
                    const int o = off[u*nline + l], s = sz[u*nline + l];
                    if (s > 0) {
                        double fr, fi;
                        line_den<S>(B, G, bo, o, s, fr, fi);
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
                    const double2 f = F[fo + (long long)p*nu + u];
                    const double2 wv = Wl[u];
                    ar += f.x*wv.x - f.y*wv.y;
                    ai += f.x*wv.y + f.y*wv.x;
                }
                const double2 A = warp_sum2(ar, ai);
                if (lane == 0) {
                    A_out[(long long)e*np + p] = A;
                    Psum += A.x*A.x + A.y*A.y;
                }
            }
        }

        if (lane == 0) {
            if (q_out) qacc += w[e]*Psum;
            p_out[e] = Psum;
        }
    }
    if (lane == 0 && q_out) atomicAdd(q_out, qacc);
}

template <int S>
__global__ void dense_grad_kernel_n(
    const double2* __restrict__ F, const double2* __restrict__ B,
    const double2* __restrict__ C, const double* __restrict__ G,
    const int* __restrict__ off, const int* __restrict__ sz, int nline,
    const double* __restrict__ w, const double2* __restrict__ A,
    const double2* __restrict__ D, int ne, int nu, int nparam, int np,
    double* __restrict__ dc_part, double* __restrict__ dg_part)
{
    for (int u = threadIdx.x; u < nu; u += blockDim.x) {
        const double2 c = C[u];
        double dcr = 0.0, dci = 0.0;
        double dgr[S];

        for (int l = 0; l < nline; l++) {
            const int o = off[u*nline + l], s = sz[u*nline + l];
            const int has = s > 0;
            #pragma unroll
            for (int i = 0; i < S; i++) { dgr[i] = 0.0; }

            for (int e = blockIdx.x; e < ne; e += gridDim.x) {
                const double we = w[e];
                const long long bo = (long long)e * nparam;
                const long long fo = (long long)e * (long long)np * nu;
                const long long ao = (long long)e * np;
                const double2 dv = D[(long long)e*nu + u];
                const double tr = dv.x, ti = dv.y;

                double2 bv[S];
                double orr = 1.0, oi = 0.0;
                if (has) {
                    double dr, di;
                    line_den<S>(B, G, bo, o, s, dr, di);
                    #pragma unroll
                    for (int i = 0; i < S; i++)
                        if (i < s) bv[i] = B[bo+o+i];
                    const double dn = dr*dr + di*di;
                    orr = (dv.x*dr + dv.y*di) / dn;
                    oi = (dv.y*dr - dv.x*di) / dn;
                }

                double gwr = 0.0, gwi = 0.0;
                for (int p = 0; p < np; p++) {
                    const double2 a = A[ao+p];
                    const double2 f = F[fo + (long long)p*nu + u];
                    gwr += 2.0*we*( f.x*a.x + f.y*a.y);
                    gwi += 2.0*we*( f.x*a.y - f.y*a.x);
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

struct Args {
    const double *fr, *btr, *cr, *gtr; const int *off, *sz; const double *w;
    int ne, nu, nparam, np, nline, nblk;
    double *q_out, *p_out, *a_out, *d_out, *dc_part, *dg_part, *dc_out, *dg_out;
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
            F, B, C, G, a.off, a.sz, a.w, a.ne, a.nu, a.nparam, a.np,
            A, a.p_out, a.q_out);
    if (mode != 1) {
        cudaMemsetAsync(a.dg_part, 0,
                        (size_t)a.nblk * a.nparam * sizeof(double), a.s);
        int gb = a.nu <= 128 ? 128 : 256;
        dense_grad_kernel<S><<<a.nblk, gb, 0, a.s>>>(
            F, B, C, G, a.off, a.sz, a.w, A, a.ne, a.nu, a.nparam, a.np,
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
            F, B, C, G, a.off, a.sz, a.nline, a.w, a.ne, a.nu, a.nparam, a.np,
            A, a.p_out, a.q_out, D);
    if (mode != 1) {
        cudaMemsetAsync(a.dg_part, 0,
                        (size_t)a.nblk * a.nparam * sizeof(double), a.s);
        int gb = a.nu <= 128 ? 128 : 256;
        dense_grad_kernel_n<S><<<a.nblk, gb, 0, a.s>>>(
            F, B, C, G, a.off, a.sz, a.nline, a.w, A, D, a.ne, a.nu, a.nparam,
            a.np, a.dc_part, a.dg_part);
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
    const int* off, const int* sz, const double* w,
    int ne, int nu, int nparam, int np, int nline, int nblk, int mode, int maxs,
    double* q_out, double* p_out, double* a_out, double* d_out,
    double* dc_part, double* dg_part,
    double* dc_out, double* dg_out, cudaStream_t s)
{
    Args a{fr, btr, cr, gtr, off, sz, w, ne, nu, nparam, np, nline, nblk,
           q_out, p_out, a_out, d_out, dc_part, dg_part, dc_out, dg_out, s};
    dispatch(a, mode, maxs, false);
}

extern "C" void launch_dense_factor_n(
    const double* fr, const double* btr, const double* cr, const double* gtr,
    const int* off, const int* sz, const double* w,
    int ne, int nu, int nparam, int np, int nline, int nblk, int mode, int maxs,
    double* q_out, double* p_out, double* a_out, double* d_out,
    double* dc_part, double* dg_part,
    double* dc_out, double* dg_out, cudaStream_t s)
{
    Args a{fr, btr, cr, gtr, off, sz, w, ne, nu, nparam, np, nline, nblk,
           q_out, p_out, a_out, d_out, dc_part, dg_part, dc_out, dg_out, s};
    dispatch(a, mode, maxs, true);
}
