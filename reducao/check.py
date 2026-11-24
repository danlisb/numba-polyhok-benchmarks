"""Confere as somas das duas versoes contra np.sum (comparacao exata, inteiros).
Uso: python3 check.py
"""
import os
import sys

import numpy as np

RAIZ = os.path.dirname(os.path.abspath(__file__))
dados = np.fromfile(os.path.join(RAIZ, "data", "dados.bin"), dtype=np.int32)
esperado = int(dados.sum(dtype=np.int64))

ok = True
for versao in ("numba", "polyhok"):
    caminho = os.path.join(RAIZ, "out", f"{versao}.txt")
    if not os.path.exists(caminho):
        print(f"{versao}: sem resultados (rode a versao antes)")
        continue
    with open(caminho) as f:
        valores = [int(v) for v in f.read().split()]
    if all(v == esperado for v in valores):
        print(f"{versao}: OK ({valores[0]})")
    else:
        ok = False
        print(f"{versao}: DIVERGE (obtido {valores}, esperado {esperado})")

sys.exit(0 if ok else 1)
