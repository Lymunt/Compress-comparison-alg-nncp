CXX ?= c++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic
RUNS ?= 1

.PHONY: all run nncp compare
all: build/hw1_ex9

build/hw1_ex9: src/arithmetic.cpp src/experiment.cpp src/arithmetic.hpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) src/arithmetic.cpp src/experiment.cpp -o $@

run: all
	./build/hw1_ex9 datasets build/run-new --runs $(RUNS)

nncp:
	$(MAKE) -C third_party/nncp

compare: all nncp
	./build/hw1_ex9 datasets build/run-new --nncp third_party/nncp/nncp --profile default --threads 1 --runs $(RUNS)
