.PHONY: all clean run run-markdown

CXX = clang++
CXXFLAGS = -std=c++17 -O3 -Wall -Wno-deprecated-declarations
FRAMEWORKS = -framework Metal -framework Foundation -framework Accelerate

all: phase2_bench

phase2_bench: phase2_bench.mm hybrid_kernels.metal
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) $< -o $@

run: phase2_bench
	./phase2_bench

run-markdown: phase2_bench
	./phase2_bench --markdown

clean:
	rm -f phase2_bench
