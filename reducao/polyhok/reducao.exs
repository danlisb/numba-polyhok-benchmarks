# Reducao (soma) em PolyHok, portada de numba/reduction_cuda.py, que estende para varios
# blocos o exemplo oficial do numba-cuda (tests/doc_examples/test_reduction.py).
#
# Mesma estrutura da versao Numba: 1024 threads por bloco, vetor int de 1024 posicoes em
# memoria compartilhada, reducao em arvore com tid % (2*s), cada bloco grava sua soma
# parcial e o host relanca o kernel sobre as parciais ate restar um valor (sem atomicos).
#
# Mede duas coisas, com aquecimento antes (compilacao JIT fora da medicao):
#   total: copia host->GPU + kernels + copia do resultado
#   gpu:   dados ja na GPU; so kernels + copia do resultado
#
# Uso: mix run reducao.exs [REPS]

require PolyHok

PolyHok.defmodule Reducao do
  defk array_sum(data, partial, n) do
    __shared__ shr[1024]

    tid = threadIdx.x
    i = blockIdx.x * blockDim.x + threadIdx.x

    if i < n do
      shr[tid] = data[i]
    else
      shr[tid] = 0
    end

    __syncthreads()

    s = 1

    while s < blockDim.x do
      # ~>> e o operador de resto (%) da PolyHok
      if (tid ~>> (2 * s)) == 0 do
        shr[tid] = shr[tid] + shr[tid + s]
      end

      s = s * 2
      __syncthreads()
    end

    if tid == 0 do
      partial[blockIdx.x] = shr[0]
    end
  end

  # Soma um vetor que ja esta na GPU; devolve um inteiro no host
  def reduzir(d, n) when n <= 1 do
    PolyHok.get_gnx(d) |> Nx.to_flat_list() |> hd()
  end

  def reduzir(d, n) do
    block = 1024
    nb = div(n + block - 1, block)
    parcial = PolyHok.new_gnx({nb}, {:s, 32})

    PolyHok.spawn(
      &Reducao.array_sum/3,
      {nb, 1, 1},
      {block, 1, 1},
      [d, parcial, n]
    )

    reduzir(parcial, nb)
  end

  # Copia o vetor para a GPU e soma
  def soma(dados), do: reduzir(PolyHok.new_gnx(dados), Nx.size(dados))
end

reps =
  case System.argv() do
    [arg] -> String.to_integer(arg)
    _ -> 10
  end

raiz = Path.expand("..", __DIR__)
out_dir = Path.join(raiz, "out")
File.mkdir_p!(out_dir)

dados = File.read!(Path.join([raiz, "data", "dados.bin"])) |> Nx.from_binary({:s, 32})
n = Nx.size(dados)

# POLYHOK_DEBUG=1 mostra o codigo CUDA gerado para o kernel
if System.get_env("POLYHOK_DEBUG"), do: PolyHok.set_debug_logs(true)

# Aquecimento: a primeira chamada compila o kernel
Reducao.soma(dados)
PolyHok.set_debug_logs(false)

medir = fn f ->
  resultados =
    for _ <- 1..reps do
      t0 = System.monotonic_time()
      r = f.()
      t1 = System.monotonic_time()
      {System.convert_time_unit(t1 - t0, :native, :microsecond) / 1000, r}
    end

  ms = resultados |> Enum.map(&elem(&1, 0)) |> Enum.sort()
  {resultados |> List.last() |> elem(1), hd(ms), Enum.at(ms, div(length(ms), 2))}
end

{r_total, min_t, med_t} = medir.(fn -> Reducao.soma(dados) end)
d_dados = PolyHok.new_gnx(dados)
{r_gpu, min_g, med_g} = medir.(fn -> Reducao.reduzir(d_dados, n) end)

File.write!(Path.join(out_dir, "polyhok.txt"), "#{r_total}\n#{r_gpu}\n")

f = fn x -> :erlang.float_to_binary(x, decimals: 3) end
IO.puts("versao\tN\ttotal_min_ms\ttotal_mediana_ms\tgpu_min_ms\tgpu_mediana_ms")
IO.puts("PolyHok\t#{n}\t#{f.(min_t)}\t#{f.(med_t)}\t#{f.(min_g)}\t#{f.(med_g)}")
