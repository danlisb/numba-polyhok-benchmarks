"""Reducao (soma) em Numba, estendida para varios blocos.

Ponto de partida: exemplo oficial do numba-cuda
(numba/cuda/tests/doc_examples/test_reduction.py, copiado sem alteracoes em
test_reduction_original.py). O original so funciona com um bloco de 1024
elementos, porque grava o resultado em data[0].

O que mudou em relacao ao original (o resto do kernel e identico):
  1. cada bloco grava sua soma parcial em partial[blockIdx.x], em vez de data[0];
  2. threads alem do fim do vetor carregam 0 na memoria compartilhada, para que
     todas as threads do bloco cheguem aos syncthreads();
  3. o tamanho n e passado como parametro (no original: len(data));
  4. o host relanca o kernel sobre as somas parciais ate restar um valor
     (sem operacoes atomicas).
"""
import numpy as np
from numba import cuda
from numba.cuda.types import int32

BLOCK = 1024  # mesmo tamanho de bloco (e de memoria compartilhada) do original


@cuda.jit
def array_sum(data, partial, n):
    tid = cuda.threadIdx.x
    i = cuda.grid(1)

    # Declare an array in shared memory
    shr = cuda.shared.array(BLOCK, int32)
    if i < n:
        shr[tid] = data[i]
    else:
        shr[tid] = 0

    # Ensure writes to shared memory are visible
    # to all threads before reducing
    cuda.syncthreads()

    s = 1
    while s < cuda.blockDim.x:
        if tid % (2 * s) == 0:
            # Stride by `s` and add
            shr[tid] += shr[tid + s]
        s *= 2
        cuda.syncthreads()

    # After the loop, the zeroth element contains the block's sum
    if tid == 0:
        partial[cuda.blockIdx.x] = shr[tid]


def reduzir(d_dados):
    """Soma um vetor que ja esta na GPU; devolve um int no host."""
    d, n = d_dados, d_dados.size
    while n > 1:
        nb = (n + BLOCK - 1) // BLOCK
        parcial = cuda.device_array(nb, dtype=np.int32)
        array_sum[nb, BLOCK](d, parcial, n)
        d, n = parcial, nb
    return int(d.copy_to_host()[0])


def soma(dados):
    """Copia o vetor para a GPU e soma (equivale ao fluxo do exemplo original)."""
    return reduzir(cuda.to_device(dados))
