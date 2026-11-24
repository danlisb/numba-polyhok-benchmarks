
__device__
float cnd(float d)
{
	float a1 = 0.31938153;
	float a2 = (- 0.356563782);
	float a3 = 1.781477937;
	float a4 = (- 1.821255978);
	float a5 = 1.330274429;
	float rsqrt2pi = 0.3989422804014327;
	float k = (1.0 / (1.0 + (0.2316419 * fabs(d))));
	float ret_val = ((rsqrt2pi * exp((((- 0.5) * d) * d))) * (k * (a1 + (k * (a2 + (k * (a3 + (k * (a4 + (k * a5))))))))));
if((d > 0.0))
{
	ret_val = (1.0 - ret_val);
}

return (ret_val);
}


__global__
void black_scholes_kernel(float *call_result, float *put_result, float *s, float *x, float *t, float r, float v, int n)
{
	int i = (threadIdx.x + (blockIdx.x * blockDim.x));
if((i < n))
{
	float sqrt_t = sqrt(t[i]);
	float d1 = ((log((s[i] / x[i])) + ((r + ((0.5 * v) * v)) * t[i])) / (v * sqrt_t));
	float d2 = (d1 - (v * sqrt_t));
	float cndd1 = cnd(d1);
	float cndd2 = cnd(d2);
	float exp_rt = exp((((- 1.0) * r) * t[i]));
	call_result[i] = ((s[i] * cndd1) - ((x[i] * exp_rt) * cndd2));
	put_result[i] = (((x[i] * exp_rt) * (1.0 - cndd2)) - (s[i] * (1.0 - cndd1)));
}

}

