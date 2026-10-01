# UMMA Phase 0 — Hardware Characterization Build
# Metal shaders compiled at runtime if metallib not available

.PHONY: all clean run run-bw run-dispatch run-slc run-markdown

CXX = clang++
CXXFLAGS = -std=c++17 -O3 -Wall -Wno-deprecated-declarations
FRAMEWORKS = -framework Metal -framework Foundation -framework Accelerate

all: phase0_bench

# Compile driver (shaders compiled at runtime from .metal source)
phase0_bench: phase0_bench.mm uma_kernels.metal
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) $< -o $@

run: phase0_bench
	./phase0_bench

run-bw: phase0_bench
	./phase0_bench --bw

run-dispatch: phase0_bench
	./phase0_bench --dispatch

run-slc: phase0_bench
	./phase0_bench --slc

run-markdown: phase0_bench
	./phase0_bench --markdown

clean:
	rm -f phase0_bench uma_kernels.air uma_kernels.metallib
