# Integração de Monte Carlo: Numba × PolyHok

Ponto de partida: exemplo oficial do numba-cuda
(`numba/cuda/tests/doc_examples/test_montecarlo.py`, numba-cuda 0.30.4),
copiado sem alterações em `numba/test_montecarlo_original.py`. Integra f(x) = 1/x
em [1, 2] e [2, 3] com amostras uniformes do gerador xoroshiro128+ no device.

## Estrutura

| Caminho | O que é |
|---|---|
| `numba/test_montecarlo_original.py` | Exemplo original, sem alterações |
| `numba/montecarlo_cuda.py` | Mesmo kernel, função e redução; `mc_integrate` dividida em duas partes |
| `numba/run.py` | Valida e mede a versão Numba |
| `polyhok/montecarlo.exs` | Port para a PolyHok |
| `polyhok/codigo_gerado.cu` | CUDA que a PolyHok gera (kernel com f(x) = 1/x e redução) |
| `gen_data.py` | Estados iniciais do gerador (seed 42), amostras de referência e tamanho de bloco |
| `check.py` | Confere o gerador e f(amostra) bit a bit, e as integrais |
| `bench.sh` | 3 rodadas de 30 repetições das duas versões + validação |

## O que mudou na versão Numba

Só a organização de `mc_integrate`: no original, ela cria os estados do gerador (na CPU),
aloca `out` copiando um vetor de zeros, lança o kernel e reduz. Aqui, estados e `out` são
criados antes, e a função medida (`integrar`) só lança o kernel e reduz. O kernel, a função
integrada, o `@cuda.reduce` e o fator de normalização são os do original. Com 1 milhão de
amostras, a versão separada dá os mesmos valores do original (0,6929643 e 0,4054021).

## O que foi preciso escrever na PolyHok

1. **Gerador xoroshiro128+.** A PolyHok não tem gerador no device, e seus tipos são só
   `int` (32 bits com sinal), `float` e `double`: não há inteiro de 64 bits nem sem sinal.
   Cada estado de 64 bits virou dois `int`, e cada operação de 64 bits (soma com vai-um,
   xor, rotação, deslocamento) é feita sobre as duas metades com os operadores bit a bit
   da PolyHok (`<<<`, `>>>`, `&&&`, `|||`, `+++`). Os deslocamentos mascaram o operando antes,
   para o C gerado não ter comportamento indefinido com `int` negativo.
2. **Redução.** O original usa `@cuda.reduce` (biblioteca do Numba). A PolyHok tem um
   equivalente, o esqueleto `reduce` (`lib/poly_hok/ske.ex`), mas na versão usada (PolyHok 0.1.1,
   branch `main` de lups-ufpel/poly_hok, de 19/11/2026) esse arquivo está inteiro comentado.
   Então a soma usa o kernel de redução em árvore
   do programa `../reducao`, em `float`. **Os algoritmos de redução são diferentes**, por
   isso o tempo da amostragem é medido separado.
3. **Função como parâmetro.** `f(x) = 1/x` é passada ao kernel com `PolyHok.phok`
   (kernel de ordem superior), como o original passa `func`.

Mesmo tamanho de bloco nas duas versões: o que o `forall` do Numba escolhe pela API de
ocupação (1024 nesta GPU), gravado em `data/tpb.txt`.

## Tipos

No original, os limites são inteiros do Python (`int64`), e amostra `float32` × `int64` dá
`float64` no Numba: a normalização da amostra e f(x) = 1/x são calculadas em `float64` e só
o resultado vai para o vetor `float32`. O port faz o mesmo (limites e `samp` em `double`;
a PolyHok infere `double` para f).

## Validação

Partindo dos mesmos estados iniciais:

- **gerador:** as amostras uniformes da PolyHok são idênticas bit a bit às do Numba
  (16.777.216 de 16.777.216);
- **f(amostra):** os valores de f antes da redução também são idênticos bit a bit;
- **integrais:** dentro de 0,001 dos valores do Maple (critério do teste original). As duas
  versões diferem em até ~8e-8, só por causa da ordem da soma (`@cuda.reduce` × redução em árvore).

## Medidas

Com aquecimento antes (JIT fora da medição) e estados já na GPU:

- `amostragem`: só o kernel que sorteia e avalia f;
- `total`: kernel + redução + resultado no host.

## Como rodar

Pré-requisitos e instalação: veja o [README principal](../README.md).

```bash
source ~/venv-numba/bin/activate
python gen_data.py       # 2^24 amostras (~3 s para criar os estados na CPU)
bash bench.sh
```

Para ver o CUDA gerado pela PolyHok: `POLYHOK_DEBUG=1 mix run montecarlo.exs 3` (na pasta `polyhok`).
