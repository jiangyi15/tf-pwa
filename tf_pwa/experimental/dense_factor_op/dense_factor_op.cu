#include <cuda_runtime.h>

#define NWARP 8
#define NTHREAD 256

// den[u,e] = sum_i g[off[u]+i] * B[e, off[u]+i]
// A_p[e]   = sum_u F[p,e,u] * C[u] / den[u,e]          (one den, all p)
// P[e]     = sum_p |A_p[e]|^2

template <int S, int P>
__global__ void dense_fwd_kernel(
    const double2* __restrict__ F,   // [ne * P * nu]
    const double2* __restrict__ B,   // [ne * nparam]
    const double2* __restrict__ C,   // [nu]
    const double2* __restrict__ G,   // [nparam]
    const int* __restrict__ off, const int* __restrict__ sz,
    const double* __restrict__ w, int ne, int nu, int nparam,
    double2* __restrict__ A_out,     // [ne * P]
    double* __restrict__ p_out,      // [ne]
    double* __restrict__ q_out)
{
    const int lane  = threadIdx.x & 31;
    const int wid   = threadIdx.x >> 5;
    const int gwarp = blockIdx.x * NWARP + wid;
    const int nwarp = gridDim.x * NWARP;
    double qacc = 0.0;

    for (int e = gwarp; e < ne; e += nwarp) {
        const long long bo = (long long)e * nparam;
        const long long fo = (long long)e * P * nu;
        double ar[P], ai[P];
        #pragma unroll
        for (int p = 0; p < P; p++) { ar[p] = 0.0; ai[p] = 0.0; }

        for (int u = lane; u < nu; u += 32) {
            const int o = off[u], s = sz[u];
            const double2 c = C[u];
            double dr = 0.0, di = 0.0;
            #pragma unroll
            for (int i = 0; i < S; i++) {
                if (i < s) {
                    const double2 b = B[bo+o+i];
                    const double2 gg = G[o+i];
                    dr += gg.x*b.x - gg.y*b.y;
                    di += gg.x*b.y + gg.y*b.x;
                }
            }
            const double den = dr*dr + di*di;
            const double ir =  dr/den, ii = -di/den;
            const double wr = c.x*ir - c.y*ii;
            const double wi = c.x*ii + c.y*ir;
            #pragma unroll
            for (int p = 0; p < P; p++) {
                const double2 f = F[fo + (long long)p*nu + u];
                ar[p] += f.x*wr - f.y*wi;
                ai[p] += f.x*wi + f.y*wr;
            }
        }
        #pragma unroll
        for (int p = 0; p < P; p++) {
            for (int o = 16; o > 0; o >>= 1) {
                ar[p] += __shfl_down_sync(0xffffffffu, ar[p], o);
                ai[p] += __shfl_down_sync(0xffffffffu, ai[p], o);
            }
            ar[p] = __shfl_sync(0xffffffffu, ar[p], 0);
            ai[p] = __shfl_sync(0xffffffffu, ai[p], 0);
        }
        if (lane == 0) {
            const long long ao = (long long)e * P;
            double Psum = 0.0;
            #pragma unroll
            for (int p = 0; p < P; p++) {
                A_out[ao+p] = make_double2(ar[p], ai[p]);
                Psum += ar[p]*ar[p] + ai[p]*ai[p];
            }
            if (q_out) qacc += w[e]*Psum;
            p_out[e] = Psum;
        }
    }
    if (lane == 0 && q_out) atomicAdd(q_out, qacc);
}

template <int S, int P>
__global__ void dense_grad_kernel(
    const double2* __restrict__ F,
    const double2* __restrict__ B,
    const double2* __restrict__ C,
    const double2* __restrict__ G,
    const int* __restrict__ off, const int* __restrict__ sz,
    const double* __restrict__ w,
    const double2* __restrict__ A,   // [ne * P]
    int ne, int nu, int nparam,
    double* __restrict__ dc_part, double* __restrict__ dg_part)
{
    const int u = threadIdx.x;
    if (u >= nu) return;
    const int o = off[u], s = sz[u];
    const double2 c = C[u];
    double dcr = 0.0, dci = 0.0;
    double dgr[S], dgi[S];
    #pragma unroll
    for (int i = 0; i < S; i++) { dgr[i] = dgi[i] = 0.0; }

    for (int e = blockIdx.x; e < ne; e += gridDim.x) {
        const double we = w[e];
        const long long bo = (long long)e * nparam;
        const long long fo = (long long)e * P * nu;
        const long long ao = (long long)e * P;
        double2 bv[S];
        double dr = 0.0, di = 0.0;
        #pragma unroll
        for (int i = 0; i < S; i++) {
            if (i < s) {
                bv[i] = B[bo+o+i];
                const double2 gg = G[o+i];
                dr += gg.x*bv[i].x - gg.y*bv[i].y;
                di += gg.x*bv[i].y + gg.y*bv[i].x;
            }
        }
        double gwr = 0.0, gwi = 0.0;
        #pragma unroll
        for (int p = 0; p < P; p++) {
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
                dgi[i] += gdi*bv[i].x - gdr*bv[i].y;
            }
        }
    }
    dc_part[(long long)blockIdx.x*2*nu + 2*u]     = dcr;
    dc_part[(long long)blockIdx.x*2*nu + 2*u + 1] = dci;
    #pragma unroll
    for (int i = 0; i < S; i++) {
        if (i < s) {
            dg_part[(long long)blockIdx.x*2*nparam + 2*(o+i)]     = dgr[i];
            dg_part[(long long)blockIdx.x*2*nparam + 2*(o+i) + 1] = dgi[i];
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

struct Args {
    const double *fr, *btr, *cr, *gtr; const int *off, *sz; const double *w;
    int ne, nu, nparam, np, nblk;
    double *q_out, *p_out, *a_out, *dc_part, *dg_part, *dc_out, *dg_out;
    cudaStream_t s;
};

template <int S, int P>
static void run(const Args& a, int mode) {
    const double2* F = reinterpret_cast<const double2*>(a.fr);
    const double2* B = reinterpret_cast<const double2*>(a.btr);
    const double2* C = reinterpret_cast<const double2*>(a.cr);
    const double2* G = reinterpret_cast<const double2*>(a.gtr);
    double2* A = reinterpret_cast<double2*>(a.a_out);
    int nb_f = a.ne < 4096 ? (a.ne + NWARP - 1) / NWARP : 2048;
    if (nb_f < 1) nb_f = 1;
    if (mode != 2)
        dense_fwd_kernel<S,P><<<nb_f, NTHREAD, 0, a.s>>>(
            F, B, C, G, a.off, a.sz, a.w, a.ne, a.nu, a.nparam,
            A, a.p_out, a.q_out);
    if (mode != 1) {
        int gb = a.nu <= 128 ? 128 : 256;
        dense_grad_kernel<S,P><<<a.nblk, gb, 0, a.s>>>(
            F, B, C, G, a.off, a.sz, a.w, A, a.ne, a.nu, a.nparam,
            a.dc_part, a.dg_part);
        reduce_k<<<(2*a.nu+255)/256, 256, 0, a.s>>>(
            a.dc_part, a.nblk, 2*a.nu, 0, 2*a.nu, a.dc_out);
        reduce_k<<<(2*a.nparam+255)/256, 256, 0, a.s>>>(
            a.dg_part, a.nblk, 2*a.nparam, 0, 2*a.nparam, a.dg_out);
    }
}

template <int P>
static void run_s(const Args& a, int mode, int maxs) {
    switch (maxs) {
        case 1: run<1,P>(a, mode); break;
        case 2: run<2,P>(a, mode); break;
        case 3: run<3,P>(a, mode); break;
        case 4: run<4,P>(a, mode); break;
        case 5: run<5,P>(a, mode); break;
        case 6: run<6,P>(a, mode); break;
        case 7: run<7,P>(a, mode); break;
        case 8: run<8,P>(a, mode); break;
        default: run<16,P>(a, mode); break;
    }
}

extern "C" void launch_dense_factor(
    const double* fr, const double* btr, const double* cr, const double* gtr,
    const int* off, const int* sz, const double* w,
    int ne, int nu, int nparam, int np, int nblk, int mode, int maxs,
    double* q_out, double* p_out, double* a_out,
    double* dc_part, double* dg_part,
    double* dc_out, double* dg_out, cudaStream_t s)
{
    Args a{fr, btr, cr, gtr, off, sz, w, ne, nu, nparam, np, nblk,
           q_out, p_out, a_out, dc_part, dg_part, dc_out, dg_out, s};
    switch (np) {
        case 1: run_s<1>(a, mode, maxs); break;
        case 2: run_s<2>(a, mode, maxs); break;
        case 3: run_s<3>(a, mode, maxs); break;
        case 4: run_s<4>(a, mode, maxs); break;
        case 5: run_s<5>(a, mode, maxs); break;
        case 6: run_s<6>(a, mode, maxs); break;
        case 7: run_s<7>(a, mode, maxs); break;
        default: run_s<8>(a, mode, maxs); break;
    }
}
