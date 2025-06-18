# Bishop
Bishop is a work-in-progress neural operator library in C++.

## Building

The project uses CMake and requires Eigen and gnuplot. A basic build can be
performed with:

```bash
mkdir build && cd build
cmake .. && make
```

Running `bishop_example` will train a tiny 1‑D neural operator that learns to
double the frequency of a sine wave and then saves a plot:

```bash
./bishop_example
```

The resulting image is stored as `plot.png` (not included in this repository).
