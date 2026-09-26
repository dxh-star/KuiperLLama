
#include <device_launch_parameters.h>
#include <cub/block/block_reduce.cuh>
#include "layernorm_kernel.cuh"
namespace kernel {
template <int32_t BLOCK_DIM>
static __global__ void layernorm_f32(const float* in, const float* gamma, const float* beta,
                                     float* out, int size, float eps) {
  const int tid = threadIdx.x;
  constexpr int pack_size = 4;
  const int pack_num = size / pack_size;
  const int pack_off = pack_size * pack_num;
  float sum = 0.0f;
  const float4* in_pack = reinterpret_cast<const float4*>(in);
  for (int i = tid; i < pack_num; i += blockDim.x) { float4 v=*(in_pack+i); sum+=v.x+v.y+v.z+v.w; }
  for (int i = pack_off + tid; i < size; i += blockDim.x) sum += in[i];
  using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ float shared_mean; __shared__ float shared_inv_std;
  sum = BlockReduce(temp).Sum(sum);
  if (tid == 0) shared_mean = sum / static_cast<float>(size);
  __syncthreads();
  float var_sum = 0.0f;
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 v=*(in_pack+i);
    float dx=v.x-shared_mean; var_sum+=dx*dx; dx=v.y-shared_mean; var_sum+=dx*dx;
    dx=v.z-shared_mean; var_sum+=dx*dx; dx=v.w-shared_mean; var_sum+=dx*dx;
  }
  for (int i = pack_off + tid; i < size; i += blockDim.x) { float dx=in[i]-shared_mean; var_sum+=dx*dx; }
  __shared__ typename BlockReduce::TempStorage temp2;
  var_sum = BlockReduce(temp2).Sum(var_sum);
  if (tid == 0) shared_inv_std = rsqrtf(var_sum/static_cast<float>(size)+eps);
  __syncthreads();
  const float4* gamma_pack = reinterpret_cast<const float4*>(gamma);
  const float4* beta_pack = reinterpret_cast<const float4*>(beta);
  float4* out_pack = reinterpret_cast<float4*>(out);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_val=*(in_pack+i); float4 g_val=*(gamma_pack+i); float4 b_val=*(beta_pack+i);
    *(out_pack+i)=make_float4(g_val.x*(in_val.x-shared_mean)*shared_inv_std+b_val.x,
                              g_val.y*(in_val.y-shared_mean)*shared_inv_std+b_val.y,
                              g_val.z*(in_val.z-shared_mean)*shared_inv_std+b_val.z,
                              g_val.w*(in_val.w-shared_mean)*shared_inv_std+b_val.w);
  }
  for (int i = pack_off + tid; i < size; i += blockDim.x)
    out[i] = gamma[i]*(in[i]-shared_mean)*shared_inv_std+beta[i];
}
void layernorm_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& gamma,
                         const tensor::Tensor& beta, const tensor::Tensor& output, void* stream) {
  CHECK(!input.is_empty()); CHECK(!gamma.is_empty()); CHECK(!beta.is_empty()); CHECK(!output.is_empty());
  CHECK(input.device_type()==base::DeviceType::kDeviceCUDA && gamma.device_type()==base::DeviceType::kDeviceCUDA &&
        beta.device_type()==base::DeviceType::kDeviceCUDA && output.device_type()==base::DeviceType::kDeviceCUDA);
  const float eps=1e-5f;
  const int32_t size=static_cast<int32_t>(input.size());
  const float* in_ptr=input.ptr<float>();
  const float* gamma_ptr=gamma.ptr<float>();
  const float* beta_ptr=beta.ptr<float>();
  float* out_ptr=const_cast<float*>(output.ptr<float>());
  constexpr int threads_num=128;
  if (stream) {
    cudaStream_t s=static_cast<cudaStream_t>(stream);
    layernorm_f32<128><<<1,threads_num,0,s>>>(in_ptr,gamma_ptr,beta_ptr,out_ptr,size,eps);
  } else {
    layernorm_f32<128><<<1,threads_num>>>(in_ptr,gamma_ptr,beta_ptr,out_ptr,size,eps);
  }
}
}  // namespace kernel
