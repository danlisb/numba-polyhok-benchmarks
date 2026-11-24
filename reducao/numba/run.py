"""Mede a reducao em Numba.

Duas medidas, com aquecimento antes (compilacao JIT fora da medicao):
  total: copia host->GPU + kernels + copia do resultado
  gpu:   dados ja na GPU; so kernels + copia do resultado
Uso: python3 run.py [REPS]
"""
import os
import sys
import time
import warnings

import numpy as np
from numba import cuda

# As ultimas passadas tem poucos blocos (ex.: grid de 1); o aviso e esperado.
warnings.filterwarnings("ignore", message="Grid size")

from reduction_cuda import reduzir, soma

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(RAIZ, "out")
os.makedirs(OUT, exist_ok=True)

dados = np.fromfile(os.path.join(RAIZ, "data", "dados.bin"), dtype=np.int32)
n = len(dados)

soma(dados)  # aquecimento


def medir(f, arg):
    tempos, r = [], None
    for _ in range(REPS):
        t0 = time.perf_counter()
        r = f(arg)
        tempos.append((time.perf_counter() - t0) * 1000)
    tempos.sort()
    return r, tempos[0], tempos[len(tempos) // 2]


r_total, min_t, med_t = medir(soma, dados)
d_dados = cuda.to_device(dados)
r_gpu, min_g, med_g = medir(reduzir, d_dados)

with open(os.path.join(OUT, "numba.txt"), "w") as f:
    f.write(f"{r_total}\n{r_gpu}\n")

print("versao\tN\ttotal_min_ms\ttotal_mediana_ms\tgpu_min_ms\tgpu_mediana_ms")
print(f"Numba\t{n}\t{min_t:.3f}\t{med_t:.3f}\t{min_g:.3f}\t{med_g:.3f}")
