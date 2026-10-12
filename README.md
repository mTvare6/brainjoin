# JIT Compiler for Brainjoin

Fork of [tsoding/bfjit](https://github.com/tsoding/bfjit)

Brainjoin is an extension of brainfuck implementing join and fork semantics inspired by [this comment](https://www.reddit.com/r/programming/comments/1wptfjd/comment/pbzdfvk/). It introduces three additional tokens: `()#`.
The characters within `()` run in a child thread whose pointer starts one cell to the right. The nth `#` blocks until the nth child finishes. See `examples/` for examples.

It is also written to be very fast combining mutliple techniques found across several implementations with re-encoding optimizations, including general forms of patterns found in [rayfuck](https://github.com/mTvare6/rayfuck) (about 216,000x faster than the base compiler)

## Quick Start

```console
$ cc -O3 -DNDEBUG -pthread -o brainjoin bfjit.c
$ ./brainjoin ./examples/fork-join.bf
```

## Benchmarks

The measurements were done on Intel Core Ultra 9 185H with the same core.
Scores listed are relative to brainjoin.

### Time

| Program | Brainjoin | tsoding | BF-JIT | Tritium | bffsree |
|---|---:|---:|---:|---:|---:|
| Mandelbrot | **327.5 ± 4.7 ms** | 757.7 ± 3.9 ms (2.31×) | 339.5 ± 2.9 ms (1.04×) | 336.7 ± 2.7 ms (1.03×) | 831.3 ± 8.7 ms (2.54×) |
| Hanoi | **3.456 ± 0.113 ms** | 1,025.9 ± 17.0 ms (296.87×) | 13.225 ± 0.428 ms (3.83×) | 10.362 ± 0.528 ms (3.00×) | 6.736 ± 0.170 ms (1.95×) |
| Golden ratio | 6.367 ± 0.133 ms | 11.101 ± 0.262 ms (1.74×) | 8.902 ± 0.236 ms (1.40×) | **6.290 ± 0.614 ms (0.99×)** | 8.235 ± 0.126 ms (1.29×) |
| Long loops | **5.231 ± 0.519 ms** | 2,135.0 ± 56.9 ms (408.18×) | 7.060 ± 0.589 ms (1.35×) | 5.292 ± 0.335 ms (1.01×) | 38.201 ± 1.235 ms (7.30×) |
| Reduced Rayfuck | **403.7 ± 2.5 ms** | >120 s (>297×) | 2,237.2 ± 10.2 ms (5.54×) | >120 s (>297×) | >120 s (>297×) |

as mean ± std

### Instructions

| Program | Brainjoin | tsoding | BF-JIT | Tritium | bffsree |
|---|---:|---:|---:|---:|---:|
| Mandelbrot | **3.794B** | 5.538B (1.46×) | 4.738B (1.25×) | 5.300B (1.40×) | 13.147B (3.47×) |
| Hanoi | **43.45M** | 8.684B (199.86×) | 158.77M (3.65×) | 154.16M (3.55×) | 94.89M (2.18×) |
| Golden ratio | **63.01M** | 97.61M (1.55×) | 94.66M (1.50×) | 71.73M (1.14×) | 87.79M (1.39×) |
| Long loops | **68.17M** | 9.871B (144.81×) | 103.73M (1.52×) | 107.41M (1.58×) | 668.43M (9.81×) |
| Reduced Rayfuck | **3.017B** | >1.450T (>480.5×) | 25.120B (8.33×) | >411.766B (>136.5×) | >1.831T (>606.9×) |

Versions: [tsoding](https://github.com/tsoding/bfjit/commit/ec07f941a8e59eb71b2193efbfaf79516ff3759b), [BF-JIT](https://github.com/danthedaniel/BF-JIT/tree/b366a64a9d83cbed1b985560be35e03c410f1eb9), [Tritium](https://github.com/rdebath/Brainfuck/tree/525d346c006ea30dfc847ae3b32ed44d44fa9925), [bffsree](https://github.com/sreekotay/bffsree/tree/fbe53a84327d63cee861c621d24e9dbe9ad097e6)

Programs: [Mandelbrot](https://github.com/danthedaniel/BF-JIT/blob/b366a64a9d83cbed1b985560be35e03c410f1eb9/tests/programs/mandelbrot.bf), [Hanoi](https://github.com/sreekotay/bffsree/blob/fbe53a84327d63cee861c621d24e9dbe9ad097e6/BFBench-1.4/hanoi.b), [Golden ratio](https://github.com/sreekotay/bffsree/blob/fbe53a84327d63cee861c621d24e9dbe9ad097e6/BFBench-1.4/golden.b), [Long loops](https://github.com/sreekotay/bffsree/blob/fbe53a84327d63cee861c621d24e9dbe9ad097e6/BFBench-1.4/long.b), [Reduced Rayfuck](https://github.com/danthedaniel/BF-JIT/blob/b366a64a9d83cbed1b985560be35e03c410f1eb9/tests/programs/ray-small.bf).
