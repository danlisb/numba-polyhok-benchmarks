# Numba × PolyHok

Programas de GPU escritos em [Numba](https://github.com/NVIDIA/numba-cuda), tirados de
repositórios oficiais e portados para a [PolyHok](https://github.com/lups-ufpel/poly_hok),
uma DSL para GPU em Elixir. A ideia é comparar as duas linguagens rodando o mesmo programa.

## Programas

| Programa | De onde vem o código Numba | Pasta |
|---|---|---|
| **Black-Scholes** | [numba-examples](https://github.com/numba/numba-examples) | [`blackscholes`](blackscholes) |
| **Redução (soma)** | exemplos do [numba-cuda](https://github.com/NVIDIA/numba-cuda) | [`reducao`](reducao) |
| **Integração de Monte Carlo** | exemplos do [numba-cuda](https://github.com/NVIDIA/numba-cuda) | [`montecarlo`](montecarlo) |

Cada pasta tem o código Numba original, o port para a PolyHok, um gerador de entradas
comum às duas versões, um validador e um script de medição. O README de cada uma explica
o que foi preciso mudar ou escrever.

## Resultados

Mediana de 30 execuções numa RTX 5060, sem contar a compilação, com os dados já na GPU.
Tamanhos: 4 milhões de opções no Black-Scholes, 33 milhões de inteiros na redução e
16 milhões de amostras no Monte Carlo.

| Programa | O que foi medido | Numba | PolyHok |
|---|---|---:|---:|
| Black-Scholes | só o kernel | 3,4 ms | 3,4 ms |
| Redução | kernel do exemplo | 1,9 ms | 3,7 ms |
| Redução | redução genérica | 0,6 ms | 1,2 ms |
| Monte Carlo | só a amostragem | 1,7 ms | 2,4 ms |
| Monte Carlo | amostragem + soma | 1,9 ms | 3,2 ms |

**As duas versões dão o mesmo resultado.** No Black-Scholes, as saídas são idênticas bit a bit.
Na redução, a soma é exata. No Monte Carlo, as amostras também são idênticas, e a integral
difere só na 8ª casa decimal, pela ordem da soma.

### O que explica as diferenças

- **Black-Scholes:** Numba e PolyHok têm o mesmo desempenho. Medindo só o kernel, as duas
  versões levam cerca de 3,4 ms; a diferença entre elas é menor que a variação entre rodadas.
- **Redução, kernel do exemplo:** o código CUDA gerado é igual ao do Numba. A diferença vem,
  pelo menos em parte, de como a PolyHok lança cada kernel: ela carrega o módulo e espera o
  kernel terminar a cada chamada. O restante ainda não foi isolado.
- **Redução genérica:** compara o `@cuda.reduce` do Numba com o `reduce` da PolyHok enviado pelo
  prof. Andre Du Bois (o esqueleto `reduce` que vem na PolyHok 0.1.1, em `ske.ex`, está todo
  comentado). As duas recebem a operação como função, mas usam algoritmos diferentes. O `reduce`
  deixa a PolyHok três vezes mais rápida que o port do exemplo, e o Numba ainda é cerca de duas
  vezes mais rápido.
- **Monte Carlo:** a PolyHok não tem gerador aleatório nem inteiros de 64 bits, então o gerador
  foi escrito com pares de inteiros de 32 bits, e a amostragem fica uns 40% mais lenta. A soma
  final usa as reduções genéricas acima (`@cuda.reduce` no Numba, `reduce` na PolyHok).

## Limitações da PolyHok encontradas

- Não há inteiros de 64 bits, inteiros sem sinal nem gerador de números aleatórios.
- O esqueleto `reduce`, que monta uma redução a partir de uma função, está com o código todo
  comentado na versão usada (0.1.1, de 19/11/2026). Foi usado no lugar o `reduce` enviado pelo
  prof. Andre Du Bois.
- O verificador de tipos rejeita alguns usos de `double`: um número sozinho
  (`var a double = 0.5`), a comparação com um número (`d > 0.0`) e a gravação de um
  `double` num vetor `float`.
- **Bug:** ao reatribuir, dentro de um `if`, uma variável criada com `var`, a PolyHok declara
  uma variável nova no código C e o valor se perde, sem nenhum erro.

Os contornos usados estão no README de cada programa.

## Como rodar

Testado no Windows com WSL 2 (Ubuntu 24.04). Placas RTX 50xx precisam de CUDA 12.8 ou mais novo.

**1. Ambiente** (uma vez só)

```bash
# No PowerShell do Windows
wsl --install -d Ubuntu-24.04

# No Ubuntu: CUDA, Erlang e ferramentas
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt-get update
sudo apt-get install -y build-essential git cmake python3-venv cuda-toolkit-12-9 erlang erlang-dev

# Elixir 1.17 (o do apt é antigo demais)
wget https://github.com/elixir-lang/elixir/releases/download/v1.17.3/elixir-otp-25.zip
sudo mkdir -p /opt/elixir && sudo unzip -q elixir-otp-25.zip -d /opt/elixir
echo 'export PATH=/usr/local/cuda/bin:/opt/elixir/bin:$PATH' >> ~/.bashrc && source ~/.bashrc
mix local.hex --force && mix local.rebar --force

# Numba (com NumPy abaixo de 2.4, que quebra o numba-cuda atual)
python3 -m venv ~/venv-numba
~/venv-numba/bin/pip install "numba-cuda[cu12]" "numpy<2.4"
```

**2. Rodar um programa** (exemplo com `reducao`)

```bash
source ~/venv-numba/bin/activate
cd reducao
(cd polyhok && mix deps.get && mix compile)   # só na primeira vez
python gen_data.py                            # gera as entradas
bash bench.sh                                 # mede as duas versões e valida
```

## Licenças

O código Numba original vem do numba-examples e do numba-cuda, ambos sob licença BSD.
Os arquivos de licença estão na pasta `numba/` de cada programa.
