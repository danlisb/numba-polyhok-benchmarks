// ===== [PolyHok] Generated code for kernel 'array_sum' =====
__global__
void array_sum(int *data, int *partial, int n)
{
__shared__ int shr[1024];
	int tid = threadIdx.x;
	int i = ((blockIdx.x * blockDim.x) + threadIdx.x);
if((i < n))
{
	shr[tid] = data[i];
}
else{
	shr[tid] = 0;
}
__syncthreads();
	int s = 1;
while((s < blockDim.x)){
if(((tid % (2 * s)) == 0))
{
	shr[tid] = (shr[tid] + shr[(tid + s)]);
}
	s = (s * 2);
__syncthreads();
}
if((tid == 0))
{
	partial[blockIdx.x] = shr[0];
}
}
