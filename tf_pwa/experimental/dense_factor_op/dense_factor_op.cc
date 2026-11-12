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
    // fs/bs share one element type: the stored real block, whatever
    // store_dtype lowered it to.  The arithmetic inside stays double.
    .Input("fs: T")              // [ne, nf], the real F block
    .Input("bs: T")              // [ne, nb], the real B block
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
    .Attr("T: {double, float, half} = DT_DOUBLE")
    .Attr("max_basis: int = 16")
    .Attr("np: int = 1");        // projections per event
REGISTER_OP("DenseFactorGradRi")
    .Input("fs: T")
    .Input("bs: T")
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
    .Attr("T: {double, float, half} = DT_DOUBLE")
    .Attr("max_basis: int = 16")
    .Attr("np: int = 1");

// The basis-index pair: F and B are the stored *basis* columns of
// method: basis_index, and each reduced column is a sparse exact transform
// of them, stored as CSR (exact row bounds, no padding).  Nothing is rebuilt
// in Python.
REGISTER_OP("DenseFactorFwdBi")
    // fs/bs share one element type: the stored basis block, whatever
    // store_dtype lowered it to.  The arithmetic inside stays double.
    .Input("fs: T")              // [ne, nf], the real F basis block
    .Input("bs: T")              // [ne, nb], the real B basis block
    .Input("c: complex128")      // [nu]
    .Input("g: float64")         // [nparam], real by construction
    .Input("off: int32")         // [nu * nline]
    .Input("sz: int32")          // [nu * nline]
    .Input("mu: int32")          // [nu], full -> reduced F column
    .Input("mi: int32")          // [nparam], full -> reduced B column
    .Input("fre: int32")         // [n_basis_f], F basis real part column
    .Input("fim: int32")         // [n_basis_f], F basis imaginary part column
    .Input("bre: int32")         // [n_basis_b], B basis real part column
    .Input("bim: int32")         // [n_basis_b], B basis imaginary part column
    .Input("f_ptr: int32")       // [np * nu_red + 1], CSR row offsets
    .Input("f_col: int32")       // [nnz_f], basis positions
    .Input("f_val: complex128")  // [nnz_f], weights
    .Input("b_ptr: int32")       // [nparam_red + 1], CSR row offsets
    .Input("b_col: int32")       // [nnz_b], basis positions
    .Input("b_val: complex128")  // [nnz_b], weights
    .Output("a: complex128")     // [ne * np], the amplitude
    .Output("d: complex128")     // [ne, nu]
    .Attr("T: {double, float, half} = DT_DOUBLE")
    .Attr("max_basis: int = 16")
    .Attr("np: int = 1")         // projections per event
    .Attr("terms: int = 4");     // widest transform row, unrolled
REGISTER_OP("DenseFactorGradBi")
    .Input("fs: T")
    .Input("bs: T")
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
    .Input("f_ptr: int32")
    .Input("f_col: int32")
    .Input("f_val: complex128")
    .Input("b_ptr: int32")
    .Input("b_col: int32")
    .Input("b_val: complex128")
    .Input("dl_da: complex128")  // dL/dA, [ne * np]
    .Input("d: complex128")
    .Output("dc: complex128")
    .Output("dg: float64")
    .Attr("T: {double, float, half} = DT_DOUBLE")
    .Attr("max_basis: int = 16")
    .Attr("np: int = 1")
    .Attr("terms: int = 4");

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

// ``fs``/``bs`` are void here because their element type follows the op's
// ``T`` attribute: double, float or half, dispatched by ``block_type``
// (0/1/2) inside the library.
extern "C" void launch_dense_factor_ri(
    const void*, const void*, const double*, const double*,
    const int*, const int*, const int*, const int*,
    const int*, const int*, const int*, const int*, const double*,
    int, int, int, int, int, int, int, int, int, int, int, int, int,
    double*, double*,
    double*, double*,
    double*, double*, cudaStream_t);

// ``fs``/``bs`` are void here because their element type follows the op's
// ``T`` attribute: double, float or half, dispatched by ``block_type``
// (0/1/2) inside the library.
extern "C" void launch_dense_factor_bi(
    const void*, const void*, const double*, const double*,
    const int*, const int*, const int*, const int*,
    const int*, const int*, const int*, const int*,
    const int*, const int*, const double*, const int*, const int*,
    const double*, const double*,
    int, int, int, int, int, int, int, int,
    int, int, int, int, int, int,
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
  if (cc.dtype() != DT_COMPLEX128)
    return errors::InvalidArgument("c must be complex128");
  if (gg.dtype() != DT_DOUBLE)
    return errors::InvalidArgument("g must be float64");
  if (of.dtype() != DT_INT32 || sz.dtype() != DT_INT32)
    return errors::InvalidArgument("off and sz must be int32");
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
    OP_REQUIRES(ctx, dl.dtype() == DT_COMPLEX128,
                errors::InvalidArgument("dl must be complex128"));
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
    OP_REQUIRES(ctx, dl.dtype() == DT_COMPLEX128 && d.dtype() == DT_COMPLEX128,
                errors::InvalidArgument("dl and d must be complex128"));
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
  // The two stored blocks share the op's T attribute, so TF already rejects a
  // mismatch; the explicit check keeps a clear message and pins the set of
  // element types the kernel can read.
  if (fs.dtype() != bs.dtype())
    return errors::InvalidArgument(
        "fs and bs must have the same dtype, got ", DataTypeString(fs.dtype()),
        " and ", DataTypeString(bs.dtype()));
  if (fs.dtype() != DT_DOUBLE && fs.dtype() != DT_FLOAT &&
      fs.dtype() != DT_HALF)
    return errors::InvalidArgument(
        "fs/bs must be float64, float32 or float16 (the stored real block), "
        "got ", DataTypeString(fs.dtype()));
  ne = (int)fs.dim_size(0);
  nf = (int)fs.dim_size(1);
  if (bs.dims() != 2 || bs.dim_size(0) != ne)
    return errors::InvalidArgument("bs must be [ne,nb]");
  nb = (int)bs.dim_size(1);
  if (nf < 1 || nb < 1)
    return errors::InvalidArgument("fs and bs must be non-empty");
  nu = (int)cc.NumElements();
  nparam = gg.NumElements();
  if (cc.dtype() != DT_COMPLEX128)
    return errors::InvalidArgument("c must be complex128");
  if (gg.dtype() != DT_DOUBLE)
    return errors::InvalidArgument("g must be float64");
  if (of.dtype() != DT_INT32 || sz.dtype() != DT_INT32)
    return errors::InvalidArgument("off and sz must be int32");
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

// Element type of the stored basis block -> the library's dispatch code.
static int block_type_of(const Tensor& t) {
  switch (t.dtype()) {
    case DT_DOUBLE: return 0;
    case DT_FLOAT: return 1;
    case DT_HALF: return 2;
    default: return -1;
  }
}

// Raw address of a block whose element type is only known at run time.
static const void* block_ptr(const Tensor& t) {
  switch (t.dtype()) {
    case DT_DOUBLE: return t.flat<double>().data();
    case DT_FLOAT: return t.flat<float>().data();
    case DT_HALF: return t.flat<Eigen::half>().data();
    default: return nullptr;
  }
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
    // -1 would silently dispatch the double kernel on a narrow block
    OP_REQUIRES(ctx, block_type_of(fs) >= 0,
                errors::InvalidArgument("unsupported stored-block dtype"));
    auto s = ctx->eigen_gpu_device().stream();
    launch_dense_factor_ri(
        block_ptr(fs), block_ptr(bs),
        reinterpret_cast<const double*>(cc.flat<complex128>().data()),
        gg.flat<double>().data(),
        of.flat<int>().data(), sz.flat<int>().data(),
        mu.flat<int>().data(), mi.flat<int>().data(),
        fre.flat<int>().data(), fim.flat<int>().data(),
        bre.flat<int>().data(), bim.flat<int>().data(), nullptr,
        ne, nu, nu_red, (int)nparam, (int)nparam_red, np_, nf, nb, nline,
        nblk, /*mode=*/1, maxb_, block_type_of(fs),
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
    OP_REQUIRES(ctx, dl.dtype() == DT_COMPLEX128 && d.dtype() == DT_COMPLEX128,
                errors::InvalidArgument("dl and d must be complex128"));
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
    // -1 would silently dispatch the double kernel on a narrow block
    OP_REQUIRES(ctx, block_type_of(fs) >= 0,
                errors::InvalidArgument("unsupported stored-block dtype"));
    auto s = ctx->eigen_gpu_device().stream();
    launch_dense_factor_ri(
        block_ptr(fs), block_ptr(bs),
        reinterpret_cast<const double*>(cc.flat<complex128>().data()),
        gg.flat<double>().data(),
        of.flat<int>().data(), sz.flat<int>().data(),
        mu.flat<int>().data(), mi.flat<int>().data(),
        fre.flat<int>().data(), fim.flat<int>().data(),
        bre.flat<int>().data(), bim.flat<int>().data(),
        reinterpret_cast<const double*>(dl.flat<complex128>().data()),
        ne, nu, nu_red, (int)nparam, (int)nparam_red, np_, nf, nb, nline,
        nblk, /*mode=*/2, maxb_, block_type_of(fs),
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

/* Shapes of the basis-index pair.
 *
 * The stored blocks hold the *basis* columns, so ``fre``/``bre`` give their
 * widths (``n_basis_f``/``n_basis_b``).  The transform is CSR:
 * ``f_ptr`` has ``np * nu_red + 1`` offsets, ``f_col``/``f_val`` the same
 * number of entries; ``b_ptr``/``b_col``/``b_val`` likewise for the
 * ``nparam_red`` B columns.  ``np`` is an attribute because the flattened F
 * block does not carry the projection axis. */
static Status check_shapes_bi(
    const Tensor& fs, const Tensor& bs, const Tensor& cc, const Tensor& gg,
    const Tensor& of, const Tensor& sz, const Tensor& mu, const Tensor& mi,
    const Tensor& fre, const Tensor& fim, const Tensor& bre, const Tensor& bim,
    const Tensor& f_ptr, const Tensor& f_col, const Tensor& f_val,
    const Tensor& b_ptr, const Tensor& b_col, const Tensor& b_val,
    int np,
    int& ne, int& nu, int& nu_red, int& nline, int64& nparam,
    int64& nparam_red, int& nf, int& nb, int& n_basis_f, int& n_basis_b) {
  if (fs.dims() != 2) return errors::InvalidArgument("fs must be [ne,nf]");
  // The two stored blocks share the op's T attribute, so TF already rejects a
  // mismatch; the explicit check keeps a clear message and pins the set of
  // element types the kernel can read.
  if (fs.dtype() != bs.dtype())
    return errors::InvalidArgument(
        "fs and bs must have the same dtype, got ", DataTypeString(fs.dtype()),
        " and ", DataTypeString(bs.dtype()));
  if (fs.dtype() != DT_DOUBLE && fs.dtype() != DT_FLOAT &&
      fs.dtype() != DT_HALF)
    return errors::InvalidArgument(
        "fs/bs must be float64, float32 or float16 (the stored basis block), "
        "got ", DataTypeString(fs.dtype()));
  ne = (int)fs.dim_size(0);
  nf = (int)fs.dim_size(1);
  if (bs.dims() != 2 || bs.dim_size(0) != ne)
    return errors::InvalidArgument("bs must be [ne,nb]");
  nb = (int)bs.dim_size(1);
  if (nf < 1 || nb < 1)
    return errors::InvalidArgument("fs and bs must be non-empty");
  nu = (int)cc.NumElements();
  nparam = gg.NumElements();
  if (cc.dtype() != DT_COMPLEX128)
    return errors::InvalidArgument("c must be complex128");
  if (gg.dtype() != DT_DOUBLE)
    return errors::InvalidArgument("g must be float64");
  if (of.dtype() != DT_INT32 || sz.dtype() != DT_INT32)
    return errors::InvalidArgument("off and sz must be int32");
  if (np < 1) return errors::InvalidArgument("np must be >= 1");
  if (nu < 1) return errors::InvalidArgument("c must be non-empty");
  if (fre.dims() != 1) return errors::InvalidArgument("fre must be 1-D");
  n_basis_f = (int)fre.NumElements();
  if (n_basis_f < 1) return errors::InvalidArgument("fre must be non-empty");
  if (fim.NumElements() != fre.NumElements())
    return errors::InvalidArgument("fim must match fre");
  if (bre.dims() != 1) return errors::InvalidArgument("bre must be 1-D");
  n_basis_b = (int)bre.NumElements();
  if (n_basis_b < 1) return errors::InvalidArgument("bre must be non-empty");
  if (bim.NumElements() != bre.NumElements())
    return errors::InvalidArgument("bim must match bre");
  if (f_ptr.dims() != 1 || f_col.dims() != 1 || f_val.dims() != 1)
    return errors::InvalidArgument("f_ptr/f_col/f_val must be 1-D");
  const int64 ncol_f = f_ptr.NumElements() - 1;
  if (ncol_f < 1)
    return errors::InvalidArgument("f_ptr must be [ncol+1]");
  if (ncol_f % np)
    return errors::InvalidArgument("f_ptr must be [np*nu_red+1]");
  nu_red = (int)(ncol_f / np);
  if (f_col.NumElements() != f_val.NumElements())
    return errors::InvalidArgument("f_col and f_val must match");
  if (b_ptr.dims() != 1 || b_col.dims() != 1 || b_val.dims() != 1)
    return errors::InvalidArgument("b_ptr/b_col/b_val must be 1-D");
  nparam_red = b_ptr.NumElements() - 1;
  if (nparam_red < 1)
    return errors::InvalidArgument("b_ptr must be [nparam_red+1]");
  if (b_col.NumElements() != b_val.NumElements())
    return errors::InvalidArgument("b_col and b_val must match");
  if (nu_red < 1 || nu_red > nu)
    return errors::InvalidArgument("f_ptr has ", nu_red, " columns for ", nu,
                                   " chains");
  if (nparam_red < 1 || nparam_red > nparam)
    return errors::InvalidArgument("b_ptr has ", nparam_red, " columns for ",
                                   nparam, " coefficients");
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
      mu.dtype() != DT_INT32 || mi.dtype() != DT_INT32 ||
      f_ptr.dtype() != DT_INT32 || f_col.dtype() != DT_INT32 ||
      b_ptr.dtype() != DT_INT32 || b_col.dtype() != DT_INT32)
    return errors::InvalidArgument("index maps must be int32");
  if (f_val.dtype() != DT_COMPLEX128 || b_val.dtype() != DT_COMPLEX128)
    return errors::InvalidArgument("transform weights must be complex128");
  return OkStatus();
}

class DenseFactorFwdBiKernel : public OpKernel {
 public:
  explicit DenseFactorFwdBiKernel(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("max_basis", &maxb_));
    OP_REQUIRES_OK(c, c->GetAttr("np", &np_));
    OP_REQUIRES_OK(c, c->GetAttr("terms", &terms_));
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
    const Tensor& f_ptr = ctx->input(12);
    const Tensor& f_col = ctx->input(13);
    const Tensor& f_val = ctx->input(14);
    const Tensor& b_ptr = ctx->input(15);
    const Tensor& b_col = ctx->input(16);
    const Tensor& b_val = ctx->input(17);
    int ne, nu, nu_red, nline, nf, nb, n_basis_f, n_basis_b;
    int64 nparam, nparam_red;
    OP_REQUIRES_OK(ctx, check_shapes_bi(fs, bs, cc, gg, of, sz, mu, mi,
                                          fre, fim, bre, bim,
                                          f_ptr, f_col, f_val,
                                          b_ptr, b_col, b_val, np_,
                                          ne, nu, nu_red, nline,
                                          nparam, nparam_red, nf, nb,
                                          n_basis_f, n_basis_b));
    // Only the forward caches the denominator in shared memory, and only
    // when the chains do not fit in registers (nu > 128); 49152 is
    // SHMEM_LIMIT and 16 is sizeof(double2) in the .cu.
    OP_REQUIRES(ctx, (int64)nu * 16 <= 49152,
                errors::InvalidArgument(
                    "nu > 3072 exceeds the shared-memory tile"));
    OP_REQUIRES(ctx, maxb_ <= 16, errors::InvalidArgument("max_basis must be <= 16"));
    Tensor* a = nullptr; Tensor* d = nullptr;
    OP_REQUIRES(ctx, terms_ >= 1 && terms_ <= 4,
                errors::InvalidArgument("terms must be 1..4, got ", terms_));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({ne*np_}), &a));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape({ne, nu}), &d));
    const int nblk = ne < 4096 ? 128 : 2048;
    // -1 would silently dispatch the double kernel on a narrow block
    OP_REQUIRES(ctx, block_type_of(fs) >= 0,
                errors::InvalidArgument("unsupported stored-block dtype"));
    auto s = ctx->eigen_gpu_device().stream();
    launch_dense_factor_bi(
        block_ptr(fs), block_ptr(bs),
        reinterpret_cast<const double*>(cc.flat<complex128>().data()),
        gg.flat<double>().data(),
        of.flat<int>().data(), sz.flat<int>().data(),
        mu.flat<int>().data(), mi.flat<int>().data(),
        fre.flat<int>().data(), fim.flat<int>().data(),
        bre.flat<int>().data(), bim.flat<int>().data(),
        f_ptr.flat<int>().data(), f_col.flat<int>().data(),
        reinterpret_cast<const double*>(f_val.flat<complex128>().data()),
        b_ptr.flat<int>().data(), b_col.flat<int>().data(),
        reinterpret_cast<const double*>(b_val.flat<complex128>().data()),
        nullptr,
        ne, nu, nu_red, (int)nparam, (int)nparam_red, np_, nf, nb,
        terms_, nline,
        nblk, /*mode=*/1, maxb_, block_type_of(fs),
        reinterpret_cast<double*>(a->flat<complex128>().data()),
        reinterpret_cast<double*>(d->flat<complex128>().data()),
        nullptr, nullptr, nullptr, nullptr, s);
  }
  int maxb_ = 16;
  int np_ = 1;
  int terms_ = 2;
};
REGISTER_KERNEL_BUILDER(Name("DenseFactorFwdBi").Device(DEVICE_GPU),
                            DenseFactorFwdBiKernel);

class DenseFactorGradBiKernel : public OpKernel {
 public:
  explicit DenseFactorGradBiKernel(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("max_basis", &maxb_));
    OP_REQUIRES_OK(c, c->GetAttr("np", &np_));
    OP_REQUIRES_OK(c, c->GetAttr("terms", &terms_));
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
    const Tensor& f_ptr = ctx->input(12);
    const Tensor& f_col = ctx->input(13);
    const Tensor& f_val = ctx->input(14);
    const Tensor& b_ptr = ctx->input(15);
    const Tensor& b_col = ctx->input(16);
    const Tensor& b_val = ctx->input(17);
    const Tensor& dl = ctx->input(18);
    const Tensor& d = ctx->input(19);
    OP_REQUIRES(ctx, dl.dtype() == DT_COMPLEX128 && d.dtype() == DT_COMPLEX128,
                errors::InvalidArgument("dl and d must be complex128"));
    int ne, nu, nu_red, nline, nf, nb, n_basis_f, n_basis_b;
    int64 nparam, nparam_red;
    OP_REQUIRES_OK(ctx, check_shapes_bi(fs, bs, cc, gg, of, sz, mu, mi,
                                          fre, fim, bre, bim,
                                          f_ptr, f_col, f_val,
                                          b_ptr, b_col, b_val, np_,
                                          ne, nu, nu_red, nline,
                                          nparam, nparam_red, nf, nb,
                                          n_basis_f, n_basis_b));
    OP_REQUIRES(ctx, maxb_ <= 16, errors::InvalidArgument("max_basis must be <= 16"));
    OP_REQUIRES(ctx, terms_ >= 1 && terms_ <= 4,
                errors::InvalidArgument("terms must be 1..4, got ", terms_));
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
    // -1 would silently dispatch the double kernel on a narrow block
    OP_REQUIRES(ctx, block_type_of(fs) >= 0,
                errors::InvalidArgument("unsupported stored-block dtype"));
    auto s = ctx->eigen_gpu_device().stream();
    launch_dense_factor_bi(
        block_ptr(fs), block_ptr(bs),
        reinterpret_cast<const double*>(cc.flat<complex128>().data()),
        gg.flat<double>().data(),
        of.flat<int>().data(), sz.flat<int>().data(),
        mu.flat<int>().data(), mi.flat<int>().data(),
        fre.flat<int>().data(), fim.flat<int>().data(),
        bre.flat<int>().data(), bim.flat<int>().data(),
        f_ptr.flat<int>().data(), f_col.flat<int>().data(),
        reinterpret_cast<const double*>(f_val.flat<complex128>().data()),
        b_ptr.flat<int>().data(), b_col.flat<int>().data(),
        reinterpret_cast<const double*>(b_val.flat<complex128>().data()),
        reinterpret_cast<const double*>(dl.flat<complex128>().data()),
        ne, nu, nu_red, (int)nparam, (int)nparam_red, np_, nf, nb,
        terms_, nline,
        nblk, /*mode=*/2, maxb_, block_type_of(fs),
        nullptr,
        reinterpret_cast<double*>(const_cast<complex128*>(d.flat<complex128>().data())),
        dcp.flat<double>().data(), dgp.flat<double>().data(),
        reinterpret_cast<double*>(dc->flat<complex128>().data()),
        dg->flat<double>().data(), s);
  }
  int maxb_ = 16;
  int np_ = 1;
  int terms_ = 2;
};
REGISTER_KERNEL_BUILDER(Name("DenseFactorGradBi").Device(DEVICE_GPU),
                            DenseFactorGradBiKernel);
