CC=nvcc
ARCH=sm_50
COMP=compute_50
CFLAGS=-g -G -gencode arch=$(COMP),code=$(ARCH) -Wno-deprecated-gpu-targets 

rayTracer: main.cu *.h
	$(CC) main.cu $(CFLAGS) -o $@

clean:
	@rm rayTracer ||:
