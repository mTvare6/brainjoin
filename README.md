# JIT Compiler for Brainjoin

Fork of [tsoding/bfjit](https://github.com/tsoding/bfjit)

Brainjoin is an extension of brainfuck implementing join and fork semantics inspired by [this comment](https://www.reddit.com/r/programming/comments/1wptfjd/comment/pbzdfvk/). It works by introducing three additional additional tokens `()#`.
The characters within `()` a process and the child process has the pointer moved one cell right. And n^th `#` blocks until n^th child's scope ends. Check examples for examples.

It also is very fast using techniques I found across multiple implementation and re-encoding optimizations, some involving large patterns with my brainfuck code in [rayfuck](https://github.com/mTvare6/rayfuck).

## Quick Start

```console
$ cc -o brainjoin bfjit.c
$ ./brainjoin ./examples/fork-join.bf
```
