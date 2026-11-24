"""Executa o Black-Scholes original do numba-examples sem alterar o kernel.

Le as entradas de ../data, faz uma execucao de aquecimento (compilacao JIT),
grava os resultados em ../out e mede REPS execucoes de duas formas:
  total: copias host<->GPU + kernel (a funcao original black_scholes_cuda)
  gpu:   so o kernel original, com os dados ja na GPU (mesma grade: 1024 threads por bloco)
Uso: python3 run.py [REPS]
"""
import math
import os
import sys
import time

import numpy as np
from numba import cuda

# arquivo original, sem modificacoes
from blackscholes_cuda import black_scholes_cuda, black_scholes_cuda_kernel

RISKFREE = 0.02     # mesmos valores do input_generator original
VOLATILITY = 0.30

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(RAIZ, "data")
OUT = os.path.join(RAIZ, "out")
os.makedirs(OUT, exist_ok=True)

S = np.fromfile(os.path.join(DATA, "S.bin"), dtype=np.float32)
X = np.fromfile(os.path.join(DATA, "X.bin"), dtype=np.float32)
T = np.fromfile(os.path.join(DATA, "T.bin"), dtype=np.float32)
n = len(S)

# Aquecimento (a primeira chamada compila o kernel) e resultado para a validacao
call, put = black_scholes_cuda(S, X, T, RISKFREE, VOLATILITY)
call.astype(np.float32).tofile(os.path.join(OUT, "numba_call.bin"))
put.astype(np.float32).tofile(os.path.join(OUT, "numba_put.bin"))


def medir(f):
    tempos = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        f()
        tempos.append((time.perf_counter() - t0) * 1000)
    tempos.sort()
    return tempos[0], tempos[len(tempos) // 2]  # mesma definicao de mediana da versao PolyHok


min_t, med_t = medir(lambda: black_scholes_cuda(S, X, T, RISKFREE, VOLATILITY))

# So o kernel: mesma configuracao de lancamento da funcao original
d_S, d_X, d_T = cuda.to_device(S), cuda.to_device(X), cuda.to_device(T)
d_call, d_put = cuda.device_array_like(S), cuda.device_array_like(S)
blockdim = 1024
griddim = int(math.ceil(float(n) / blockdim))


def so_kernel():
    black_scholes_cuda_kernel[griddim, blockdim](
        d_call, d_put, d_S, d_X, d_T, RISKFREE, VOLATILITY)
    cuda.synchronize()


so_kernel()
min_g, med_g = medir(so_kernel)

print("versao\tN\ttotal_min_ms\ttotal_mediana_ms\tgpu_min_ms\tgpu_mediana_ms")
print(f"Numba\t{n}\t{min_t:.3f}\t{med_t:.3f}\t{min_g:.3f}\t{med_g:.3f}")
