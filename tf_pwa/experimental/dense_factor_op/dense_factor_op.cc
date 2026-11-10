#define EIGEN_USE_GPU
#include <cuda_runtime.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
using namespace tensorflow;

REGISTER_OP("DenseFactorFwd")
    .Input("f: complex128")     // [ne, np, nu]
    .Input("bt: complex128")    // [ne, nparam]
    .Input("c: complex128")     // [nu]
    .Input("g: float64")        // [nparam], real by construction
    .Input("off: int32")        // [nu * nline]
    .Input("sz: int32")         // [nu * nline]
    .Output("a: complex128")    // [ne * np], the amplitude
    .Attr("max_basis: int = 16");
REGISTER_OP("DenseFactorFwdMerge")
    .Input("f: complex128")     // [ne, np, nu]
    .Input("bt: complex128")    // [ne, nparam]
    .Input("c: complex128")     // [nu]
    .Input("g: float64")        // [nparam], real by construction
    .Input("off: int32")        // [nu * nline]
    .Input("sz: int32")         // [nu * nline]
    .Input("mu: int32")         // [nu], full -> reduced F column
    .Input("mi: int32")         // [nparam], full -> reduced B column
    .Output("a: complex128")    // [ne * np], the amplitude
    .Attr("max_basis: int = 16");
REGISTER_OP("DenseFactorGrad")
    .Input("f: complex128")
    .Input("bt: complex128")
    .Input("c: complex128")
    .Input("g: float64")
    .Input("off: int32")
    .Input("sz: int32")
    .Input("dl_da: complex128")  // dL/dA, [ne * np]
    .Output("dc: complex128")
    .Output("dg: float64")
    .Attr("max_basis: int = 16");
REGISTER_OP("DenseFactorGradMerge")
    .Input("f: complex128")
    .Input("bt: complex128")
    .Input("c: complex128")
    .Input("g: float64")
    .Input("off: int32")
    .Input("sz: int32")
    .Input("mu: int32")         // [nu], full -> reduced F column
    .Input("mi: int32")         // [nparam], full -> reduced B column
    .Input("dl_da: complex128")  // dL/dA, [ne * np]
    .Output("dc: complex128")
    .Output("dg: float64")
    .Attr("max_basis: int = 16");
REGISTER_OP("DenseFactorFwdN")
    .Input("f: complex128")
    .Input("bt: complex128")
    .Input("c: complex128")
    .Input("g: float64")
    .Input("off: int32")
    .Input("sz: int32")
    .Output("a: complex128")    // [ne * np], the amplitude
    .Output("d: complex128")    // [ne, nu]
    .Attr("max_basis: int = 16");
REGISTER_OP("DenseFactorFwdNMerge")
    .Input("f: complex128")
    .Input("bt: complex128")
    .Input("c: complex128")
    .Input("g: float64")
    .Input("off: int32")
    .Input("sz: int32")
    .Input("mu: int32")         // [nu], full -> reduced F column
    .Input("mi: int32")         // [nparam], full -> reduced B column
    .Output("a: complex128")    // [ne * np], the amplitude
    .Output("d: complex128")    // [ne, nu]
    .Attr("max_basis: int = 16");
REGISTER_OP("DenseFactorGradN")
    .Input("f: complex128")
    .Input("bt: complex128")
    .Input("c: complex128")
    .Input("g: float64")
    .Input("off: int32")
    .Input("sz: int32")
    .Input("dl_da: complex128")  // dL/dA, [ne * np]
    .Input("d: complex128")
    .Output("dc: complex128")
    .Output("dg: float64")
    .Attr("max_basis: int = 16");
REGISTER_OP("DenseFactorGradNMerge")
    .Input("f: complex128")
    .Input("bt: complex128")
    .Input("c: complex128")
    .Input("g: float64")
    .Input("off: int32")
    .Input("sz: int32")
    .Input("mu: int32")         // [nu], full -> reduced F column
    .Input("mi: int32")         // [nparam], full -> reduced B column
    .Input("dl_da: complex128")  // dL/dA, [ne * np]
    .Input("d: complex128")
    .Output("dc: complex128")
    .Output("dg: float64")
    .Attr("max_basis: int = 16");

// The real-index pair: F and B are the merged *real* blocks of
// method: real_index, and the read maps say where the real and imaginary
// part of every reduced column lives.  Nothing is rebuilt in Python.
REGISTER_OP("DenseFactorFwdRi")
    .Input("fs: float64")        // [ne, nf], the real F block
    .Input("bs: float64")        // [ne, nb], the real B block
    .Input("c: complex128")      // [nu]
    .Input("g: float64")         // [nparam], real by construction
    .Input("off: int32")         // [nu * nline]
    .Input("sz: int32")          // [nu * nline]
    .Input("mu: int32")          // [nu], full -> reduced F column
    .Input("mi: int32")          // [nparam], full -> reduced B column
    .Input("fre: int32")         // [np * nu_red], real part column
    .Input("fim: int32")         // [np * nu_red], imaginary part column
    .Input("bre: int32")         // [nparam_red], real part column
    .Input("bim: int32")         // [nparam_red], imaginary part column
    .Output("a: complex128")     // [ne * np], the amplitude
    .Output("d: complex128")     // [ne, nu]
    .Attr("max_basis: int = 16")
    .Attr("np: int = 1");        // projections per event
REGISTER_OP("DenseFactorGradRi")
    .Input("fs: float64")
    .Input("bs: float64")
    .Input("c: complex128")
    .Input("g: float64")
    .Input("off: int32")
    .Input("sz: int32")
    .Input("mu: int32")
    .Input("mi: int32")
    .Input("fre: int32")
    .Input("fim: int32")
    .Input("bre: int32")
    .Input("bim: int32")
    .Input("dl_da: complex128")  // dL/dA, [ne * np]
    .Input("d: complex128")
    .Output("dc: complex128")
    .Output("dg: float64")
    .Attr("max_basis: int = 16")
    .Attr("np: int = 1");

extern "C" void launch_dense_factor(
    const double*, const double*, const double*, const double*,
    const int*, const int*, const int*, const int*, const double*,
    int, int, int, int, int, int, int, int, int, int, int,
    double*, double*,
    double*, double*,
    double*, double*, cudaStream_t);

extern "C" void launch_dense_factor_n(
    const double*, const double*, const double*, const double*,
    const int*, const int*, const int*, const int*, const double*,
    int, int, int, int, int, int, int, int, int, int, int,
    double*, double*,
    double*, double*,
    double*, double*, cudaStream_t);

extern "C" void launch_dense_factor_ri(
    const double*, const double*, const double*, const double*,
    const int*, const int*, const int*, const int*,
    const int*, const int*, const int*, const int*, const double*,
    int, int, int, int, int, int, int, int, int, int, int, int,
    double*, double*,
    double*, double*,
    double*, double*, cudaStream_t);

/* Shapes shared by all four kernels.
 *
 * The *values* inside off/sz are device data and are not inspected, so the
 * caller must keep off[u*nline+l] + sz[u*nline+l] <= nparam and
 * sz <= max_basis; a larger sz is truncated by the unrolled basis loop. */
static Status check_shapes(const Tensor& f, const Tensor& bt,
                           const Tensor& cc, const Tensor& gg,
                           const Tensor& of, const Tensor& sz,
                           const Tensor* mu, const Tensor* mi, bool merged,
                           int& ne, int& np, int& nu, int& nu_red,
                           int& nline, int64& nparam, int64& nparam_red) {
  if (f.dims() != 3) return errors::InvalidArgument("f must be [ne,np,nu]");
  ne = (int)f.dim_size(0);
  np = (int)f.dim_size(1);
  nu_red = (int)f.dim_size(2);
  if (bt.dims() != 2 || bt.dim_size(0) != ne)
    return errors::InvalidArgument("bt must be [ne,nparam]");
  nparam_red = bt.dim_size(1);
  // c and g are the full coefficient vectors; f/bt may be reduced when the
  // basis is stored merged, and (mu, mi) map a full index to the stored one
  nu = (int)cc.NumElements();
  nparam = gg.NumElements();
  if (np < 1) return errors::InvalidArgument("np must be >= 1");
  if (nu < 1) return errors::InvalidArgument("c must be non-empty");
  if (nu_red < 1 || nu_red > nu)
    return errors::InvalidArgument("f has ", nu_red, " columns for ", nu,
                                   " chains");
  if (nparam_red < 1 || nparam_red > nparam)
    return errors::InvalidArgument("bt has ", nparam_red, " columns for ",
                                   nparam, " coefficients");
  if (!merged && (nu_red != nu || nparam_red != nparam))
    return errors::InvalidArgument(
        "an unmerged op needs the full basis; the data was reduced by a "
        "merge preprocessor - use the Merge op");
  if (of.NumElements() != sz.NumElements())
    return errors::InvalidArgument("off and sz must match");
  nline = (int)(of.NumElements() / nu);
  if (nline < 1) return errors::InvalidArgument("nline must be >= 1");
  if (of.NumElements() != (int64)nu * nline)
    return errors::InvalidArgument("off must be [nu*nline]");
  if (merged) {
    if (mu->NumElements() != (int64)nu)
      return errors::InvalidArgument("mu must be [nu]");
    if (mi->NumElements() != nparam)
      return errors::InvalidArgument("mi must be [nparam]");
  }
  return OkStatus();
}

template <bool MERGE>
class DenseFactorFwdKernel : public OpKernel {
 public:
  explicit DenseFactorFwdKernel(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("max_basis", &maxb_));
  }
  void Compute(OpKernelContext* ctx) override {
    const Tensor& f = ctx->input(0);
    const Tensor& bt = ctx->input(1);
    const Tensor& cc = ctx->input(2);
    const Tensor& gg = ctx->input(3);
    const Tensor& of = ctx->input(4);
    const Tensor& sz = ctx->input(5);
    int ne, np, nu, nu_red, nline; int64 nparam, nparam_red;
    OP_REQUIRES_OK(ctx, check_shapes(f, bt, cc, gg, of, sz,
                                     MERGE ? &ctx->input(6) : nullptr,
                                     MERGE ? &ctx->input(7) : nullptr,
                                     MERGE,
                                     ne, np, nu, nu_red, nline,
                                     nparam, nparam_red));
    OP_REQUIRES(ctx, nline == 1,
                errors::InvalidArgument(
                    "off/sz must be [nu]; use DenseFactorFwdN/GradN for "
                    "several lines"));
    // Only the forward caches the denominator in shared memory, and only
    // when the chains do not fit in registers (nu > 128); 49152 is
    // SHMEM_LIMIT and 16 is sizeof(double2) in the .cu.
    OP_REQUIRES(ctx, (int64)nu * 16 <= 49152,
                errors::InvalidArgument(
                    "nu > 3072 exceeds the shared-memory tile"));
    OP_REQUIRES(ctx, maxb_ <= 16, errors::InvalidArgument("max_basis must be <= 16"));
    Tensor* a = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({ne*np}), &a));
    const int nblk = ne < 4096 ? 128 : 2048;
    const int* mu_d = MERGE ? ctx->input(6).flat<int>().data() : nullptr;
    const int* mi_d = MERGE ? ctx->input(7).flat<int>().data() : nullptr;
    auto s = ctx->eigen_gpu_device().stream();
    launch_dense_factor(
        reinterpret_cast<const double*>(f.flat<complex128>().data()),
        reinterpret_cast<const double*>(bt.flat<complex128>().data()),
        reinterpret_cast<const double*>(cc.flat<complex128>().data()),
        gg.flat<double>().data(),
        of.flat<int>().data(), sz.flat<int>().data(),
        mu_d, mi_d, nullptr,
        ne, nu, nu_red, (int)nparam, (int)nparam_red, np, /*nline=*/1, nblk,
        /*mode=*/1, maxb_, MERGE ? 1 : 0,
        reinterpret_cast<double*>(a->flat<complex128>().data()), nullptr,
        nullptr, nullptr, nullptr, nullptr, s);
  }
  int maxb_ = 16;
};
REGISTER_KERNEL_BUILDER(Name("DenseFactorFwd").Device(DEVICE_GPU),
                            DenseFactorFwdKernel<false>);
REGISTER_KERNEL_BUILDER(Name("DenseFactorFwdMerge").Device(DEVICE_GPU),
                            DenseFactorFwdKernel<true>);

template <bool MERGE>
class DenseFactorGradKernel : public OpKernel {
 public:
  explicit DenseFactorGradKernel(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("max_basis", &maxb_));
  }
  void Compute(OpKernelContext* ctx) override {
    const Tensor& f = ctx->input(0);
    const Tensor& bt = ctx->input(1);
    const Tensor& cc = ctx->input(2);
    const Tensor& gg = ctx->input(3);
    const Tensor& of = ctx->input(4);
    const Tensor& sz = ctx->input(5);
    const Tensor& dl = ctx->input(MERGE ? 8 : 6);
    int ne, np, nu, nu_red, nline; int64 nparam, nparam_red;
    OP_REQUIRES_OK(ctx, check_shapes(f, bt, cc, gg, of, sz,
                                     MERGE ? &ctx->input(6) : nullptr,
                                     MERGE ? &ctx->input(7) : nullptr,
                                     MERGE,
                                     ne, np, nu, nu_red, nline,
                                     nparam, nparam_red));
    OP_REQUIRES(ctx, nline == 1,
                errors::InvalidArgument(
                    "off/sz must be [nu]; use DenseFactorFwdN/GradN for "
                    "several lines"));
    OP_REQUIRES(ctx, maxb_ <= 16, errors::InvalidArgument("max_basis must be <= 16"));
    OP_REQUIRES(ctx, dl.NumElements() == (int64)ne*np,
                errors::InvalidArgument("dl_da size"));
    Tensor* dc = nullptr; Tensor* dg = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({nu}), &dc));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape({nparam}), &dg));
    const int nblk = ne < 4096 ? 128 : 2048;
    Tensor dcp, dgp;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_DOUBLE,
        TensorShape({(int64)nblk*2*nu}), &dcp));
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_DOUBLE,
        TensorShape({(int64)nblk*nparam}), &dgp));
    const int* mu_d = MERGE ? ctx->input(6).flat<int>().data() : nullptr;
    const int* mi_d = MERGE ? ctx->input(7).flat<int>().data() : nullptr;
    auto s = ctx->eigen_gpu_device().stream();
    launch_dense_factor(
        reinterpret_cast<const double*>(f.flat<complex128>().data()),
        reinterpret_cast<const double*>(bt.flat<complex128>().data()),
        reinterpret_cast<const double*>(cc.flat<complex128>().data()),
        gg.flat<double>().data(),
        of.flat<int>().data(), sz.flat<int>().data(),
        mu_d, mi_d,
        reinterpret_cast<const double*>(dl.flat<complex128>().data()),
        ne, nu, nu_red, (int)nparam, (int)nparam_red, np, /*nline=*/1, nblk,
        /*mode=*/2, maxb_, MERGE ? 1 : 0,
        nullptr, nullptr,
        dcp.flat<double>().data(), dgp.flat<double>().data(),
        reinterpret_cast<double*>(dc->flat<complex128>().data()),
        dg->flat<double>().data(), s);
  }
  int maxb_ = 16;
};
REGISTER_KERNEL_BUILDER(Name("DenseFactorGrad").Device(DEVICE_GPU),
                            DenseFactorGradKernel<false>);
REGISTER_KERNEL_BUILDER(Name("DenseFactorGradMerge").Device(DEVICE_GPU),
                            DenseFactorGradKernel<true>);

template <bool MERGE>
class DenseFactorFwdNKernel : public OpKernel {
 public:
  explicit DenseFactorFwdNKernel(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("max_basis", &maxb_));
  }
  void Compute(OpKernelContext* ctx) override {
    const Tensor& f = ctx->input(0);
    const Tensor& bt = ctx->input(1);
    const Tensor& cc = ctx->input(2);
    const Tensor& gg = ctx->input(3);
    const Tensor& of = ctx->input(4);
    const Tensor& sz = ctx->input(5);
    int ne, np, nu, nu_red, nline; int64 nparam, nparam_red;
    OP_REQUIRES_OK(ctx, check_shapes(f, bt, cc, gg, of, sz,
                                     MERGE ? &ctx->input(6) : nullptr,
                                     MERGE ? &ctx->input(7) : nullptr,
                                     MERGE,
                                     ne, np, nu, nu_red, nline,
                                     nparam, nparam_red));
    // Only the forward caches the denominator in shared memory, and only
    // when the chains do not fit in registers (nu > 128); 49152 is
    // SHMEM_LIMIT and 16 is sizeof(double2) in the .cu.
    OP_REQUIRES(ctx, (int64)nu * 16 <= 49152,
                errors::InvalidArgument(
                    "nu > 3072 exceeds the shared-memory tile"));
    OP_REQUIRES(ctx, maxb_ <= 16, errors::InvalidArgument("max_basis must be <= 16"));
    Tensor* a = nullptr; Tensor* d = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({ne*np}), &a));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape({ne, nu}), &d));
    const int nblk = ne < 4096 ? 128 : 2048;
    const int* mu_d = MERGE ? ctx->input(6).flat<int>().data() : nullptr;
    const int* mi_d = MERGE ? ctx->input(7).flat<int>().data() : nullptr;
    auto s = ctx->eigen_gpu_device().stream();
    launch_dense_factor_n(
        reinterpret_cast<const double*>(f.flat<complex128>().data()),
        reinterpret_cast<const double*>(bt.flat<complex128>().data()),
        reinterpret_cast<const double*>(cc.flat<complex128>().data()),
        gg.flat<double>().data(),
        of.flat<int>().data(), sz.flat<int>().data(),
        mu_d, mi_d, nullptr,
        ne, nu, nu_red, (int)nparam, (int)nparam_red, np, nline, nblk,
        /*mode=*/1, maxb_, MERGE ? 1 : 0,
        reinterpret_cast<double*>(a->flat<complex128>().data()),
        reinterpret_cast<double*>(d->flat<complex128>().data()),
        nullptr, nullptr, nullptr, nullptr, s);
  }
  int maxb_ = 16;
};
REGISTER_KERNEL_BUILDER(Name("DenseFactorFwdN").Device(DEVICE_GPU),
                            DenseFactorFwdNKernel<false>);
REGISTER_KERNEL_BUILDER(Name("DenseFactorFwdNMerge").Device(DEVICE_GPU),
                            DenseFactorFwdNKernel<true>);

template <bool MERGE>
class DenseFactorGradNKernel : public OpKernel {
 public:
  explicit DenseFactorGradNKernel(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("max_basis", &maxb_));
  }
  void Compute(OpKernelContext* ctx) override {
    const Tensor& f = ctx->input(0);
    const Tensor& bt = ctx->input(1);
    const Tensor& cc = ctx->input(2);
    const Tensor& gg = ctx->input(3);
    const Tensor& of = ctx->input(4);
    const Tensor& sz = ctx->input(5);
    const Tensor& dl = ctx->input(MERGE ? 8 : 6);
    const Tensor& d = ctx->input(MERGE ? 9 : 7);
    int ne, np, nu, nu_red, nline; int64 nparam, nparam_red;
    OP_REQUIRES_OK(ctx, check_shapes(f, bt, cc, gg, of, sz,
                                     MERGE ? &ctx->input(6) : nullptr,
                                     MERGE ? &ctx->input(7) : nullptr,
                                     MERGE,
                                     ne, np, nu, nu_red, nline,
                                     nparam, nparam_red));
    OP_REQUIRES(ctx, maxb_ <= 16, errors::InvalidArgument("max_basis must be <= 16"));
    OP_REQUIRES(ctx, d.NumElements() == (int64)ne*nu, errors::InvalidArgument("d size"));
    OP_REQUIRES(ctx, dl.NumElements() == (int64)ne*np,
                errors::InvalidArgument("dl_da size"));
    Tensor* dc = nullptr; Tensor* dg = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({nu}), &dc));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape({nparam}), &dg));
    const int nblk = ne < 4096 ? 128 : 2048;
    Tensor dcp, dgp;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_DOUBLE,
        TensorShape({(int64)nblk*2*nu}), &dcp));
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_DOUBLE,
        TensorShape({(int64)nblk*nparam}), &dgp));
    const int* mu_d = MERGE ? ctx->input(6).flat<int>().data() : nullptr;
    const int* mi_d = MERGE ? ctx->input(7).flat<int>().data() : nullptr;
    auto s = ctx->eigen_gpu_device().stream();
    launch_dense_factor_n(
        reinterpret_cast<const double*>(f.flat<complex128>().data()),
        reinterpret_cast<const double*>(bt.flat<complex128>().data()),
        reinterpret_cast<const double*>(cc.flat<complex128>().data()),
        gg.flat<double>().data(),
        of.flat<int>().data(), sz.flat<int>().data(),
        mu_d, mi_d,
        reinterpret_cast<const double*>(dl.flat<complex128>().data()),
        ne, nu, nu_red, (int)nparam, (int)nparam_red, np, nline, nblk,
        /*mode=*/2, maxb_, MERGE ? 1 : 0,
        nullptr,
        reinterpret_cast<double*>(const_cast<complex128*>(d.flat<complex128>().data())),
        dcp.flat<double>().data(), dgp.flat<double>().data(),
        reinterpret_cast<double*>(dc->flat<complex128>().data()),
        dg->flat<double>().data(), s);
  }
  int maxb_ = 16;
};
REGISTER_KERNEL_BUILDER(Name("DenseFactorGradN").Device(DEVICE_GPU),
                            DenseFactorGradNKernel<false>);
REGISTER_KERNEL_BUILDER(Name("DenseFactorGradNMerge").Device(DEVICE_GPU),
                            DenseFactorGradNKernel<true>);

/* Shapes of the real-index pair.
 *
 * The stored blocks are already merged and real, so the reduced widths come
 * from the maps: ``fre`` holds one entry per reduced F column and projection
 * (``np * nu_red``), ``bre`` one per reduced B column.  ``np`` is an attribute
 * because the flattened F block does not carry the projection axis. */
static Status check_shapes_ri(
    const Tensor& fs, const Tensor& bs, const Tensor& cc, const Tensor& gg,
    const Tensor& of, const Tensor& sz, const Tensor& mu, const Tensor& mi,
    const Tensor& fre, const Tensor& fim, const Tensor& bre, const Tensor& bim,
    int np, int& ne, int& nu, int& nu_red, int& nline, int64& nparam,
    int64& nparam_red, int& nf, int& nb) {
  if (fs.dims() != 2) return errors::InvalidArgument("fs must be [ne,nf]");
  if (fs.dtype() != DT_DOUBLE)
    return errors::InvalidArgument("fs must be float64");
  ne = (int)fs.dim_size(0);
  nf = (int)fs.dim_size(1);
  if (bs.dims() != 2 || bs.dim_size(0) != ne)
    return errors::InvalidArgument("bs must be [ne,nb]");
  if (bs.dtype() != DT_DOUBLE)
    return errors::InvalidArgument("bs must be float64");
  nb = (int)bs.dim_size(1);
  if (nf < 1 || nb < 1)
    return errors::InvalidArgument("fs and bs must be non-empty");
  nu = (int)cc.NumElements();
  nparam = gg.NumElements();
  if (np < 1) return errors::InvalidArgument("np must be >= 1");
  if (nu < 1) return errors::InvalidArgument("c must be non-empty");
  if (fre.dims() != 1 || fre.NumElements() % np)
    return errors::InvalidArgument("fre must be [np*nu_red]");
  nu_red = (int)(fre.NumElements() / np);
  if (fim.NumElements() != fre.NumElements())
    return errors::InvalidArgument("fim must match fre");
  nparam_red = bre.NumElements();
  if (nparam_red < 1 || nparam_red > nparam)
    return errors::InvalidArgument("bre has ", nparam_red, " entries for ",
                                   nparam, " coefficients");
  if (bim.NumElements() != nparam_red)
    return errors::InvalidArgument("bim must match bre");
  if (nu_red < 1 || nu_red > nu)
    return errors::InvalidArgument("fre has ", nu_red, " columns for ", nu,
                                   " chains");
  if (mu.NumElements() != (int64)nu)
    return errors::InvalidArgument("mu must be [nu]");
  if (mi.NumElements() != nparam)
    return errors::InvalidArgument("mi must be [nparam]");
  if (of.NumElements() != sz.NumElements())
    return errors::InvalidArgument("off and sz must match");
  nline = (int)(of.NumElements() / nu);
  if (nline < 1) return errors::InvalidArgument("nline must be >= 1");
  if (of.NumElements() != (int64)nu * nline)
    return errors::InvalidArgument("off must be [nu*nline]");
  if (fre.dtype() != DT_INT32 || fim.dtype() != DT_INT32 ||
      bre.dtype() != DT_INT32 || bim.dtype() != DT_INT32 ||
      mu.dtype() != DT_INT32 || mi.dtype() != DT_INT32)
    return errors::InvalidArgument("index maps must be int32");
  return OkStatus();
}

class DenseFactorFwdRiKernel : public OpKernel {
 public:
  explicit DenseFactorFwdRiKernel(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("max_basis", &maxb_));
    OP_REQUIRES_OK(c, c->GetAttr("np", &np_));
  }
  void Compute(OpKernelContext* ctx) override {
    const Tensor& fs = ctx->input(0);
    const Tensor& bs = ctx->input(1);
    const Tensor& cc = ctx->input(2);
    const Tensor& gg = ctx->input(3);
    const Tensor& of = ctx->input(4);
    const Tensor& sz = ctx->input(5);
    const Tensor& mu = ctx->input(6);
    const Tensor& mi = ctx->input(7);
    const Tensor& fre = ctx->input(8);
    const Tensor& fim = ctx->input(9);
    const Tensor& bre = ctx->input(10);
    const Tensor& bim = ctx->input(11);
    int ne, nu, nu_red, nline, nf, nb; int64 nparam, nparam_red;
    OP_REQUIRES_OK(ctx, check_shapes_ri(fs, bs, cc, gg, of, sz, mu, mi,
                                          fre, fim, bre, bim, np_,
                                          ne, nu, nu_red, nline,
                                          nparam, nparam_red, nf, nb));
    // Only the forward caches the denominator in shared memory, and only
    // when the chains do not fit in registers (nu > 128); 49152 is
    // SHMEM_LIMIT and 16 is sizeof(double2) in the .cu.
    OP_REQUIRES(ctx, (int64)nu * 16 <= 49152,
                errors::InvalidArgument(
                    "nu > 3072 exceeds the shared-memory tile"));
    OP_REQUIRES(ctx, maxb_ <= 16, errors::InvalidArgument("max_basis must be <= 16"));
    Tensor* a = nullptr; Tensor* d = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({ne*np_}), &a));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape({ne, nu}), &d));
    const int nblk = ne < 4096 ? 128 : 2048;
    auto s = ctx->eigen_gpu_device().stream();
    launch_dense_factor_ri(
        fs.flat<double>().data(), bs.flat<double>().data(),
        reinterpret_cast<const double*>(cc.flat<complex128>().data()),
        gg.flat<double>().data(),
        of.flat<int>().data(), sz.flat<int>().data(),
        mu.flat<int>().data(), mi.flat<int>().data(),
        fre.flat<int>().data(), fim.flat<int>().data(),
        bre.flat<int>().data(), bim.flat<int>().data(), nullptr,
        ne, nu, nu_red, (int)nparam, (int)nparam_red, np_, nf, nb, nline,
        nblk, /*mode=*/1, maxb_,
        reinterpret_cast<double*>(a->flat<complex128>().data()),
        reinterpret_cast<double*>(d->flat<complex128>().data()),
        nullptr, nullptr, nullptr, nullptr, s);
  }
  int maxb_ = 16;
  int np_ = 1;
};
REGISTER_KERNEL_BUILDER(Name("DenseFactorFwdRi").Device(DEVICE_GPU),
                            DenseFactorFwdRiKernel);

class DenseFactorGradRiKernel : public OpKernel {
 public:
  explicit DenseFactorGradRiKernel(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("max_basis", &maxb_));
    OP_REQUIRES_OK(c, c->GetAttr("np", &np_));
  }
  void Compute(OpKernelContext* ctx) override {
    const Tensor& fs = ctx->input(0);
    const Tensor& bs = ctx->input(1);
    const Tensor& cc = ctx->input(2);
    const Tensor& gg = ctx->input(3);
    const Tensor& of = ctx->input(4);
    const Tensor& sz = ctx->input(5);
    const Tensor& mu = ctx->input(6);
    const Tensor& mi = ctx->input(7);
    const Tensor& fre = ctx->input(8);
    const Tensor& fim = ctx->input(9);
    const Tensor& bre = ctx->input(10);
    const Tensor& bim = ctx->input(11);
    const Tensor& dl = ctx->input(12);
    const Tensor& d = ctx->input(13);
    int ne, nu, nu_red, nline, nf, nb; int64 nparam, nparam_red;
    OP_REQUIRES_OK(ctx, check_shapes_ri(fs, bs, cc, gg, of, sz, mu, mi,
                                          fre, fim, bre, bim, np_,
                                          ne, nu, nu_red, nline,
                                          nparam, nparam_red, nf, nb));
    OP_REQUIRES(ctx, maxb_ <= 16, errors::InvalidArgument("max_basis must be <= 16"));
    OP_REQUIRES(ctx, d.NumElements() == (int64)ne*nu, errors::InvalidArgument("d size"));
    OP_REQUIRES(ctx, dl.NumElements() == (int64)ne*np_,
                errors::InvalidArgument("dl_da size"));
    Tensor* dc = nullptr; Tensor* dg = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({nu}), &dc));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape({nparam}), &dg));
    const int nblk = ne < 4096 ? 128 : 2048;
    Tensor dcp, dgp;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_DOUBLE,
        TensorShape({(int64)nblk*2*nu}), &dcp));
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_DOUBLE,
        TensorShape({(int64)nblk*nparam}), &dgp));
    auto s = ctx->eigen_gpu_device().stream();
    launch_dense_factor_ri(
        fs.flat<double>().data(), bs.flat<double>().data(),
        reinterpret_cast<const double*>(cc.flat<complex128>().data()),
        gg.flat<double>().data(),
        of.flat<int>().data(), sz.flat<int>().data(),
        mu.flat<int>().data(), mi.flat<int>().data(),
        fre.flat<int>().data(), fim.flat<int>().data(),
        bre.flat<int>().data(), bim.flat<int>().data(),
        reinterpret_cast<const double*>(dl.flat<complex128>().data()),
        ne, nu, nu_red, (int)nparam, (int)nparam_red, np_, nf, nb, nline,
        nblk, /*mode=*/2, maxb_,
        nullptr,
        reinterpret_cast<double*>(const_cast<complex128*>(d.flat<complex128>().data())),
        dcp.flat<double>().data(), dgp.flat<double>().data(),
        reinterpret_cast<double*>(dc->flat<complex128>().data()),
        dg->flat<double>().data(), s);
  }
  int maxb_ = 16;
  int np_ = 1;
};
REGISTER_KERNEL_BUILDER(Name("DenseFactorGradRi").Device(DEVICE_GPU),
                            DenseFactorGradRiKernel);
