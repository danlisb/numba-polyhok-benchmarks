#!/bin/bash
# Roda as duas versoes 3 vezes (30 repeticoes cada) e valida.
# Uso: bash bench.sh   (depois de python3 gen_data.py)
cd "$(dirname "$0")"
. ~/venv-numba/bin/activate
echo -e "versao\tN\tamostragem_min_ms\tamostragem_mediana_ms\ttotal_min_ms\ttotal_mediana_ms\tI[1,2]\tI[2,3]"
for i in 1 2 3; do
  (cd numba && python run.py 30 | tail -1)
  (cd polyhok && mix run montecarlo.exs 30 2>/dev/null | tail -1)
done
python check.py
