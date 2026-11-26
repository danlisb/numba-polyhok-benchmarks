# Reducao (soma) com o reduce generico da PolyHok enviado pelo prof. Andre Du Bois.
#
# Diferente de reducao.exs (que segue o kernel do exemplo do Numba), este usa o algoritmo
# do reduce: 64 blocos de 256 threads, cada thread acumula varios elementos num laco
# (grid-stride), arvore com enderecamento sequencial em memoria compartilhada e duas
# chamadas de kernel (parciais dos 64 blocos -> resultado). A soma e passada como funcao.
#
# Mesmas medidas de reducao.exs:
#   total: copia host->GPU + kernels + copia do resultado
#   gpu:   dados ja na GPU; so kernels + copia do resultado
#
# Uso: mix run reducao_ske.exs [REPS]

require PolyHok

PolyHok.defmodule ReducaoSke do
  # ----- reduce enviado pelo professor (so o nome do modulo no spawn foi trocado) -----
  def reduce(ref, initial, f) do
    threadsPerBlock = 256
    numberOfBlocks = 64

    size =
      case PolyHok.get_shape_gnx(ref) do
        {l, c} -> l * c
        {size} -> size
      end

    type = PolyHok.get_type_gnx(ref)
    result_gpu = PolyHok.new_gnx(Nx.tensor([[initial]], type: type))
    d_partials = PolyHok.new_gnx({numberOfBlocks}, type)

    PolyHok.spawn(&ReducaoSke.reduce_kernel/5, {numberOfBlocks, 1, 1}, {threadsPerBlock, 1, 1}, [
      ref,
      d_partials,
      f,
      initial,
      size
    ])

    PolyHok.spawn(&ReducaoSke.reduce_kernel/5, {1, 1, 1}, {threadsPerBlock, 1, 1}, [
      d_partials,
      result_gpu,
      f,
      initial,
      numberOfBlocks
    ])

    result_gpu
  end

  defk reduce_kernel(ra, rout, f, identity, n) do
    __shared__ cache[256]
    cacheIndex = threadIdx.x
    tid = threadIdx.x + blockIdx.x * blockDim.x

    temp = identity

    while tid < n do
      temp = f(ra[tid], temp)
      tid = tid + blockDim.x * gridDim.x
    end

    cache[cacheIndex] = temp
    __syncthreads()

    type i int
    i = blockDim.x / 2

    while i > 0 do
      if cacheIndex < i do
        cache[cacheIndex] = f(cache[cacheIndex + i], cache[cacheIndex])
      end

      __syncthreads()
      i = i / 2
    end

    if cacheIndex == 0 do
      rout[blockIdx.x] = cache[0]
    end
  end

  # ----- uso no benchmark -----
  def reduzir(d) do
    ReducaoSke.reduce(d, 0, PolyHok.phok(fn a, b -> a + b end))
    |> PolyHok.get_gnx()
    |> Nx.to_flat_list()
    |> hd()
  end

  def soma(dados), do: reduzir(PolyHok.new_gnx(dados))
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

# POLYHOK_DEBUG=1 mostra o codigo CUDA gerado
if System.get_env("POLYHOK_DEBUG"), do: PolyHok.set_debug_logs(true)
ReducaoSke.soma(dados)
PolyHok.set_debug_logs(false)

medir = fn f ->
  resultados =
    for _ <- 1..reps do
      t0 = System.monotonic_time()
      r = f.()
      {System.convert_time_unit(System.monotonic_time() - t0, :native, :microsecond) / 1000, r}
    end

  ms = resultados |> Enum.map(&elem(&1, 0)) |> Enum.sort()
  {resultados |> List.last() |> elem(1), hd(ms), Enum.at(ms, div(length(ms), 2))}
end

{r_total, min_t, med_t} = medir.(fn -> ReducaoSke.soma(dados) end)
d_dados = PolyHok.new_gnx(dados)
{r_gpu, min_g, med_g} = medir.(fn -> ReducaoSke.reduzir(d_dados) end)

File.write!(Path.join(out_dir, "polyhok_ske.txt"), "#{r_total}\n#{r_gpu}\n")

f = fn x -> :erlang.float_to_binary(x, decimals: 3) end
IO.puts("versao\tN\ttotal_min_ms\ttotal_mediana_ms\tgpu_min_ms\tgpu_mediana_ms")
IO.puts("PolyHok-reduce\t#{n}\t#{f.(min_t)}\t#{f.(med_t)}\t#{f.(min_g)}\t#{f.(med_g)}")
