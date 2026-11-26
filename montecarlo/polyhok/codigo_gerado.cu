// ===== [PolyHok] Generated code for kernel 'mc_integrator_kernel' =====
__device__
int lshr(int x, int k)
{
return (((x >> k) & ((1 << (32 - k)) - 1)));
}
__device__
int shl(int x, int k)
{
return (((x & ((1 << (32 - k)) - 1)) << k));
}
__device__
int add_hi(int ah, int al, int bh, int bl)
{
	int l16 = ((al & 65535) + (bl & 65535));
	int carry = lshr(((lshr(al, 16) + lshr(bl, 16)) + lshr(l16, 16)), 16);
	int m16 = (((ah & 65535) + (bh & 65535)) + carry);
	int n16 = ((lshr(ah, 16) + lshr(bh, 16)) + lshr(m16, 16));
return ((shl(n16, 16) | (m16 & 65535)));
}
__device__
int add_lo(int al, int bl)
{
	int l16 = ((al & 65535) + (bl & 65535));
	int h16 = ((lshr(al, 16) + lshr(bl, 16)) + lshr(l16, 16));
return ((shl(h16, 16) | (l16 & 65535)));
}
__device__
double anon_d09m9gj4gd(double x)
{
return ((1.0 / x));
}
__global__
void mc_integrator_kernel(float *out, int *st, double lower_lim, double upper_lim, int size)
{
	int gid = ((blockIdx.x * blockDim.x) + threadIdx.x);
if((gid < size))
{
	int b = (4 * gid);
	int s0l = st[b];
	int s0h = st[(b + 1)];
	int s1l = st[(b + 2)];
	int s1h = st[(b + 3)];
	int rl = add_lo(s0l, s1l);
	int rh = add_hi(s0h, s0l, s1h, s1l);
	int t1l = (s1l ^ s0l);
	int t1h = (s1h ^ s0h);
	st[b] = ((shl(s0h, 23) | lshr(s0l, 9)) ^ (t1l ^ shl(t1l, 14)));
	st[(b + 1)] = ((shl(s0l, 23) | lshr(s0h, 9)) ^ (t1h ^ (shl(t1h, 14) | lshr(t1l, 18))));
	st[(b + 2)] = (shl(t1h, 4) | lshr(t1l, 28));
	st[(b + 3)] = (shl(t1l, 4) | lshr(t1h, 28));
	int a = lshr(rh, 11);
	int c = (shl(rh, 21) | lshr(rl, 11));
float u = (((a * 4294967296.0) + ((lshr(c, 16) * 65536.0) + (c & 65535))) * 1.1102230246251565e-16);
double samp = ((u * (upper_lim - lower_lim)) + lower_lim);
double y = anon_d09m9gj4gd(samp);
	out[gid] = y;
}
}
// ===== [PolyHok] Generated code for kernel 'reduce_kernel' =====
__device__
float anon_9e7l4n6c94(float a, float b)
{
return ((a + b));
}
__global__
void reduce_kernel(float *ra, float *rout, float identity, int n)
{
__shared__ float cache[256];
	int cacheIndex = threadIdx.x;
	int tid = (threadIdx.x + (blockIdx.x * blockDim.x));
	float temp = identity;
while((tid < n)){
	temp = anon_9e7l4n6c94(ra[tid], temp);
	tid = (tid + (blockDim.x * gridDim.x));
}
	cache[cacheIndex] = temp;
__syncthreads();
	int i = (blockDim.x / 2);
while((i > 0)){
if((cacheIndex < i))
{
	cache[cacheIndex] = anon_9e7l4n6c94(cache[(cacheIndex + i)], cache[cacheIndex]);
}
__syncthreads();
	i = (i / 2);
}
if((cacheIndex == 0))
{
	rout[blockIdx.x] = cache[0];
}
}
