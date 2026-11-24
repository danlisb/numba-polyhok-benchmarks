"""Integracao de Monte Carlo em Numba, extraida do exemplo oficial do numba-cuda
(numba/cuda/tests/doc_examples/test_montecarlo.py, copiado sem alteracoes em
test_montecarlo_original.py).

O kernel, a funcao integrada, a reducao (@cuda.reduce) e o calculo do fator sao os do
original. A unica mudanca e que a funcao mc_integrate foi dividida em duas partes, para
que a criacao dos estados do gerador (feita na CPU) e a alocacao do vetor de saida fiquem
fora da medicao:
  - original: mc_integrate cria os estados, aloca `out` com cuda.to_device(np.zeros(...)),
    lanca o kernel e reduz;
  - aqui: estados e `out` sao criados antes; `integrar` lanca o kernel e reduz.
"""
import numba
import numpy as np
from numba import cuda
from numba.cuda.random import xoroshiro128p_uniform_float32


@cuda.jit
def mc_integrator_kernel(out, rng_states, lower_lim, upper_lim):
    """
    kernel to draw random samples and evaluate the function to
    be integrated at those sample values
    """
    size = len(out)

    gid = cuda.grid(1)
    if gid < size:
        # draw a sample between 0 and 1 on this thread
        samp = xoroshiro128p_uniform_float32(rng_states, gid)

        # normalize this sample to the limit range
        samp = samp * (upper_lim - lower_lim) + lower_lim

        # evaluate the function to be
        # integrated at the normalized
        # value of the sample
        y = func(samp)
        out[gid] = y


@cuda.reduce
def sum_reduce(a, b):
    return a + b


# define a function to integrate
@numba.jit
def func(x):
    return 1.0 / x


def amostrar(out, rng_states, lower_lim, upper_lim):
    """So o kernel de amostragem (sincroniza para poder medir)."""
    mc_integrator_kernel.forall(len(out))(out, rng_states, lower_lim, upper_lim)
    cuda.synchronize()


def integrar(out, rng_states, lower_lim, upper_lim):
    """Kernel de amostragem + reducao, como em mc_integrate do original."""
    nsamps = len(out)
    mc_integrator_kernel.forall(nsamps)(out, rng_states, lower_lim, upper_lim)
    # normalization factor to convert
    # to the average: (b - a)/(N - 1)
    factor = (upper_lim - lower_lim) / (nsamps - 1)
    return sum_reduce(out) * factor


def threads_por_bloco(nsamps, rng_states):
    """Tamanho de bloco que o forall escolhe (API de ocupacao), para usar o mesmo na PolyHok."""
    out = cuda.device_array(nsamps, dtype=np.float32)
    fa = mc_integrator_kernel.forall(nsamps)
    spec = mc_integrator_kernel.specialize(out, rng_states, 1, 2)
    return fa._compute_thread_per_block(spec)
