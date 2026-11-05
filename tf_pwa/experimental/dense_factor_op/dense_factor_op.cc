#define EIGEN_USE_GPU
#include <cuda_runtime.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
using namespace tensorflow;

extern "C" void launch_dense_factor(
    const double*, const double*, const double*, const double*,
    const int*, const int*, const double*,
    int, int, int, int, int, int, int,
    double*, double*, double*,
    double*, double*,
    double*, double*, cudaStream_t);



// ---------------------------------------------------------------------------
// Split entry points: forward emits A[e] so the gradient can reuse it.
// ---------------------------------------------------------------------------
REGISTER_OP("DenseFactorFwd")
    .Input("f: complex128")
    .Input("bt: complex128")
    .Input("c: complex128")
    .Input("g: complex128")
    .Input("off: int32")
    .Input("sz: int32")
    .Output("a: complex128")
    .Output("p: float64")
    .Attr("max_basis: int = 16");

REGISTER_OP("DenseFactorGrad")
    .Input("f: complex128")
    .Input("bt: complex128")
    .Input("c: complex128")
    .Input("g: complex128")
    .Input("off: int32")
    .Input("sz: int32")
    .Input("dl_dp: float64")
    .Input("a: complex128")
    .Output("dc: complex128")
    .Output("dg: complex128")
    .Attr("max_basis: int = 16");

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
    OP_REQUIRES(ctx, f.dims() == 3, errors::InvalidArgument("f must be [ne,np,nu]"));
    const int ne = (int)f.dim_size(0);
    const int np = (int)f.dim_size(1);
    const int nu = (int)f.dim_size(2);
    const int64 nparam = bt.dim_size(1);
    OP_REQUIRES(ctx, np <= 8, errors::InvalidArgument("np must be <= 8"));
    OP_REQUIRES(ctx, maxb_ <= 16, errors::InvalidArgument("max_basis must be <= 16"));
    Tensor* a = nullptr; Tensor* p = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({ne*np}), &a));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape({ne}), &p));
    const int nblk = ne < 4096 ? 128 : 2048;
    auto s = ctx->eigen_gpu_device().stream();
    launch_dense_factor(
        reinterpret_cast<const double*>(f.flat<complex128>().data()),
        reinterpret_cast<const double*>(bt.flat<complex128>().data()),
        reinterpret_cast<const double*>(cc.flat<complex128>().data()),
        reinterpret_cast<const double*>(gg.flat<complex128>().data()),
        of.flat<int>().data(), sz.flat<int>().data(), nullptr,
        ne, nu, (int)nparam, np, nblk, /*mode=*/1, maxb_,
        nullptr, p->flat<double>().data(),
        reinterpret_cast<double*>(a->flat<complex128>().data()),
        nullptr, nullptr, nullptr, nullptr, s);
  }
  int maxb_ = 16;
};
REGISTER_KERNEL_BUILDER(Name("DenseFactorFwd").Device(DEVICE_GPU), DenseFactorFwdKernel);

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
    const Tensor& dl = ctx->input(6);
    const Tensor& a = ctx->input(7);
    OP_REQUIRES(ctx, f.dims() == 3, errors::InvalidArgument("f must be [ne,np,nu]"));
    const int ne = (int)f.dim_size(0);
    const int np = (int)f.dim_size(1);
    const int nu = (int)f.dim_size(2);
    const int64 nparam = bt.dim_size(1);
    OP_REQUIRES(ctx, np <= 8, errors::InvalidArgument("np must be <= 8"));
    OP_REQUIRES(ctx, maxb_ <= 16, errors::InvalidArgument("max_basis must be <= 16"));
    OP_REQUIRES(ctx, a.NumElements() == (int64)ne*np, errors::InvalidArgument("a size"));
    Tensor* dc = nullptr; Tensor* dg = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({nu}), &dc));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape({nparam}), &dg));
    const int nblk = ne < 4096 ? 128 : 2048;
    Tensor dcp, dgp;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_DOUBLE,
        TensorShape({(int64)nblk*2*nu}), &dcp));
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_DOUBLE,
        TensorShape({(int64)nblk*2*nparam}), &dgp));
    auto s = ctx->eigen_gpu_device().stream();
    launch_dense_factor(
        reinterpret_cast<const double*>(f.flat<complex128>().data()),
        reinterpret_cast<const double*>(bt.flat<complex128>().data()),
        reinterpret_cast<const double*>(cc.flat<complex128>().data()),
        reinterpret_cast<const double*>(gg.flat<complex128>().data()),
        of.flat<int>().data(), sz.flat<int>().data(), dl.flat<double>().data(),
        ne, nu, (int)nparam, np, nblk, /*mode=*/2, maxb_,
        nullptr, nullptr,
        reinterpret_cast<double*>(const_cast<complex128*>(a.flat<complex128>().data())),
        dcp.flat<double>().data(), dgp.flat<double>().data(),
        reinterpret_cast<double*>(dc->flat<complex128>().data()),
        reinterpret_cast<double*>(dg->flat<complex128>().data()), s);
  }
  int maxb_ = 16;
};
REGISTER_KERNEL_BUILDER(Name("DenseFactorGrad").Device(DEVICE_GPU), DenseFactorGradKernel);
