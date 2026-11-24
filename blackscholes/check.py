"""Confere os resultados das duas versoes contra a referencia NumPy do numba-examples.

Usa o mesmo criterio do validator original (rtol=1e-5, atol=1e-5).
Uso: python3 check.py
"""
import os
import sys

import numpy as np

RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(RAIZ, "numba"))
from blackscholes import black_scholes  # referencia NumPy original

ler = lambda pasta, nome: np.fromfile(os.path.join(RAIZ, pasta, nome), dtype=np.float32)
S, X, T = (ler("data", f"{n}.bin") for n in "SXT")
ref_call, ref_put = black_scholes(S, X, T, 0.02, 0.30)

ok = True
res = {}
for versao in ("numba", "polyhok"):
    caminho = os.path.join(RAIZ, "out", f"{versao}_call.bin")
    if not os.path.exists(caminho):
        print(f"{versao}: sem resultados (rode a versao antes)")
        continue
    call, put = ler("out", f"{versao}_call.bin"), ler("out", f"{versao}_put.bin")
    res[versao] = (call, put)
    erro = max(np.abs(call - ref_call).max(), np.abs(put - ref_put).max())
    try:
        np.testing.assert_allclose(call, ref_call, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(put, ref_put, rtol=1e-5, atol=1e-5)
        print(f"{versao}: OK (maior erro absoluto {erro:.2e})")
    except AssertionError:
        ok = False
        print(f"{versao}: DIVERGE da referencia (maior erro absoluto {erro:.2e})")

# Comparacao direta entre as duas versoes (fazem as mesmas contas, nos mesmos tipos)
if len(res) == 2:
    iguais = sum(int((a == b).sum()) for a, b in zip(res["numba"], res["polyhok"]))
    total = 2 * len(S)
    dif = max(np.abs(a - b).max() for a, b in zip(res["numba"], res["polyhok"]))
    print(f"Numba x PolyHok: {iguais} de {total} valores identicos; maior diferenca {dif:.2e}")

sys.exit(0 if ok else 1)
