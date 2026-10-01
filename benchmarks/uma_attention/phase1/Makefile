# UMMA Phase 1R — GPU-Batched Multi-Head Attention Build

.PHONY: all clean run run-markdown

CXX = clang++
CXXFLAGS = -std=c++17 -O3 -Wall -Wno-deprecated-declarations
FRAMEWORKS = -framework Metal -framework Foundation -framework Accelerate

all: phase1_bench

phase1_bench: phase1_bench.mm multihead_attention.metal
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) $< -o $@

run: phase1_bench
	./phase1_bench

run-markdown: phase1_bench
	./phase1_bench --markdown

clean:
	rm -f phase1_bench
