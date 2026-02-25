all: V0CPP V0 V1 V1_transpose V2 V3 splitting vecvec

V0CPP: V0.cpp
	g++ -o V0CPP V0.cpp

CXX := g++-10
NVCC := nvcc
NVCCFLAGS := -ccbin g++-10 -std=c++14

V0: V0.cu
	$(NVCC) $(NVCCFLAGS) -o V0 V0.cu

V1: V1.cu
	$(NVCC) $(NVCCFLAGS) -o V1 V1.cu

V1_transpose: V1_transpose.cu
	$(NVCC) $(NVCCFLAGS) -o V1_transpose V1_transpose.cu

V2: V2.cu
	$(NVCC) $(NVCCFLAGS) -o V2 V2.cu

V3: V3.cu
	$(NVCC) $(NVCCFLAGS) -o V3 V3.cu

splitting: splitting.cu
	$(NVCC) $(NVCCFLAGS) -o splitting splitting.cu

vecvec: vecvec.cu
	$(NVCC) $(NVCCFLAGS) -o vecvec vecvec.cu