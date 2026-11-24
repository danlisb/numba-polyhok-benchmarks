# Black-Scholes: Numba × PolyHok

Port do Black-Scholes do repositório oficial [numba-examples](https://github.com/numba/numba-examples)
(`examples/finance/blackscholes/blackscholes_cuda.py`) para a PolyHok.

## Estrutura

| Caminho | O que é |
|---|---|
| `numba/blackscholes_cuda.py` | Código Numba **original, sem alterações** |
| `numba/blackscholes.py` | Referência NumPy original, usada na validação |
| `numba/run.py` | Executa o original e mede o tempo |
| `polyhok/blackscholes.exs` | Port para a PolyHok |
| `polyhok/codigo_gerado.cu` | CUDA que a PolyHok gera para o port |
| `gen_data.py` | Gera as mesmas entradas para as duas versões |
| `check.py` | Confere os resultados contra a referência NumPy e compara as duas versões |
| `bench.sh` | 3 rodadas de 30 repetições das duas versões + validação |

## Como foi feita a comparação

- **Mesmas entradas:** as duas versões leem os arquivos gerados por `gen_data.py`, com a mesma
  distribuição do gerador original (S de 5 a 30, X de 1 a 100, T de 0,25 a 10; R = 0,02; V = 0,30).
- **Mesma estrutura:** uma função de device (`cnd`), uma opção por thread, 1024 threads por bloco.
- **Mesmos tipos:** veja abaixo.
- **Duas medidas**, com uma execução de aquecimento antes (compilação JIT fora da medição):
  - `total`: cópias para a GPU, kernel e cópias de volta, como a função `black_scholes_cuda` original;
  - `gpu`: só o kernel, com os dados já na GPU.
- **Validação:** `check.py` usa o mesmo critério do validador original (`rtol = atol = 1e-5`) e
  compara as saídas das duas versões entre si. **As 8 milhões de saídas (call e put) são
  idênticas bit a bit.**

## Tipos

Os vetores são `float32`, mas o Numba faz quase todas as contas em `float64`: R e V são floats
do Python (`float64`), então `d1`, `d2`, `cnd`, `expRT` e as contas de call/put saem em `float64`.
Só `sqrt(T[i])` e `log(S[i] / X[i])` ficam em `float32`. (Conferido com `inspect_types`.)

O port declara os mesmos tipos (`type r double`, `var d1 double = ...` etc.). Isso pesa no
desempenho: GPUs GeForce fazem contas `float64` bem mais devagar que `float32`.

## Limitações da PolyHok encontradas no port

1. **Literal em contexto `double`:** `var a1 double = 0.31938153` dá erro de tipo (o verificador
   só aceita literal sozinho em contexto `float`). As constantes de `cnd` foram escritas direto
   na expressão, onde o literal é aceito e o CUDA gerado já o trata como `double`.
2. **Comparação de `double` com literal:** `d > 0.0` dá erro de tipo. Foi escrito `d > d * 0.0`
   (mesma condição para `d` finito; o tipo da expressão vem do primeiro operando).
3. **Reatribuição de variável declarada com `var` (bug):** com `var ret_val double = ...` e
   `ret_val = 1.0 - ret_val` dentro de um `if`, a PolyHok gera `double ret_val = 1.0 - ret_val;`
   dentro do `if`, ou seja, declara uma variável nova e a reatribuição se perde, sem erro.
   O resultado sai errado (a validação acusou). Contorno: `type ret_val double` + atribuição comum.
4. **Sem conversão implícita de `double` para `float`:** gravar uma expressão `double` num vetor
   `float` dá erro de tipo; o valor passa por uma variável `double` antes.

## Como rodar

Pré-requisitos e instalação: veja o [README principal](../README.md).

```bash
source ~/venv-numba/bin/activate
python gen_data.py 4000000    # 4 milhões de opções, maior tamanho do benchmark original
bash bench.sh                 # 3 rodadas de 30 repetições das duas versões + validação
```

Para ver o CUDA gerado: `POLYHOK_DEBUG=1 mix run blackscholes.exs 3` (na pasta `polyhok`).
