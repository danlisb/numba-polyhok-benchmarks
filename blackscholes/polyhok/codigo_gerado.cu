// ===== [PolyHok] Generated code for kernel 'black_scholes_kernel' =====
__device__
double cnd(double d)
{
double k = (1.0 / (1.0 + (0.2316419 * fabs(d))));
	double ret_val = ((0.3989422804014327 * exp((((- 0.5) * d) * d))) * (k * (0.31938153 + (k * ((- 0.356563782) + (k * (1.781477937 + (k * ((- 1.821255978) + (k * 1.330274429))))))))));
if((d > (d * 0.0)))
{
	ret_val = (1.0 - ret_val);
}
return (ret_val);
}
__global__
void black_scholes_kernel(float *call_result, float *put_result, float *s, float *x, float *t, double r, double v, int n)
{
	int i = (threadIdx.x + (blockIdx.x * blockDim.x));
if((i < n))
{
float sqrt_t = sqrt(t[i]);
double d1 = ((log((s[i] / x[i])) + ((r + ((0.5 * v) * v)) * t[i])) / (v * sqrt_t));
double d2 = (d1 - (v * sqrt_t));
double cndd1 = cnd(d1);
double cndd2 = cnd(d2);
double exp_rt = exp((((- 1.0) * r) * t[i]));
double c = ((s[i] * cndd1) - ((x[i] * exp_rt) * cndd2));
double p = (((x[i] * exp_rt) * (1.0 - cndd2)) - (s[i] * (1.0 - cndd1)));
	call_result[i] = c;
	put_result[i] = p;
}
}
