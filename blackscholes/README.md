# Black-Scholes: Numba × PolyHok

Port do Black-Scholes do repositório oficial [numba-examples](https://github.com/numba/numba-examples)
(`examples/finance/blackscholes/blackscholes_cuda.py`, commit `c423f54`) para a PolyHok.

## Estrutura

| Caminho | O que é |
|---|---|
| `numba/blackscholes_cuda.py` | Código Numba **original, sem alterações** |
| `numba/blackscholes.py` | Referência NumPy original, usada na validação |
| `numba/run.py` | Executa o original e mede o tempo |
| `polyhok/blackscholes.exs` | Port para a PolyHok |
| `gen_data.py` | Gera as mesmas entradas para as duas versões |
| `check.py` | Confere os resultados contra a referência NumPy |
| `bench.sh` | 3 rodadas de 30 repetições das duas versões + validação |
| `polyhok/codigo_gerado_referencia.cu` | CUDA gerado pela PolyHok para o port |

## Como foi feita a comparação

- **Mesmas entradas:** as duas versões leem os arquivos gerados por `gen_data.py`, com a mesma
  distribuição do gerador original (S de 5 a 30, X de 1 a 100, T de 0,25 a 10; R = 0,02; V = 0,30).
- **Mesma estrutura:** uma função de device (`cnd`), uma opção por thread, 1024 threads por bloco.
- **O que é medido:** cópia das entradas para a GPU, execução do kernel e cópia dos resultados
  de volta, como na função `black_scholes_cuda` original.
- **Compilação fora da medição:** as duas versões fazem uma execução de aquecimento
  antes de medir, para que a compilação JIT não entre no tempo.
- **Validação:** `check.py` usa o mesmo critério do validador original (`rtol = atol = 1e-5`).

## Como rodar

Pré-requisitos e instalação: veja o [README principal](../README.md).

```bash
python3 gen_data.py 4000000   # 4 milhões de opções, maior tamanho do benchmark original
bash bench.sh                 # 3 rodadas de 30 repetições das duas versões + validação
```

Para ver o código CUDA gerado pela PolyHok, acrescente `PolyHok.set_debug_logs(true)`
antes do aquecimento em `blackscholes.exs`.

## Ponto de atenção: precisão

Os vetores são `float32` nas duas versões, mas nenhuma delas faz tudo em precisão simples:

- **Numba:** os literais (`0.5`, `1.0`) e os escalares R e V são `float64`.
- **PolyHok:** R e V viram `float`, mas o código CUDA gerado escreve os literais como `0.5` e `1.0`,
  que em C são `double`, então parte das contas também é feita em precisão dupla.

A diferença entre as duas fica só em R e V. Vale mencionar isso no trabalho.
