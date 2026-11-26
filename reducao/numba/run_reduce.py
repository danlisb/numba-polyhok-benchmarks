"""Mede a reducao pronta da biblioteca do Numba (@cuda.reduce) com os mesmos dados.

Serve de comparacao para o reduce generico da PolyHok (polyhok/reducao_ske.exs): os dois
sao reducoes genericas, que recebem a operacao como funcao.
Mesmas medidas de run.py:
  total: copia host->GPU + reducao + resultado no host
  gpu:   dados ja na GPU; so reducao + resultado no host
Uso: python3 run_reduce.py [REPS]
"""
import os
import sys
import time
import warnings

import numpy as np
from numba import cuda

# O @cuda.reduce lanca grades pequenas nas ultimas passadas; o aviso e esperado.
warnings.filterwarnings("ignore", message="Grid size")


@cuda.reduce
def sum_reduce(a, b):
    return a + b


REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(RAIZ, "out")
os.makedirs(OUT, exist_ok=True)

dados = np.fromfile(os.path.join(RAIZ, "data", "dados.bin"), dtype=np.int32)
n = len(dados)


def soma(x):
    return int(sum_reduce(cuda.to_device(x)))


def reduzir(d):
    return int(sum_reduce(d))


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

with open(os.path.join(OUT, "numba_reduce.txt"), "w") as f:
    f.write(f"{r_total}\n{r_gpu}\n")

print("versao\tN\ttotal_min_ms\ttotal_mediana_ms\tgpu_min_ms\tgpu_mediana_ms")
print(f"Numba-reduce\t{n}\t{min_t:.3f}\t{med_t:.3f}\t{min_g:.3f}\t{med_g:.3f}")
