CC=nvcc
ARCH=sm_50
COMP=compute_50
CFLAGS=-gencode arch=$(COMP),code=$(ARCH) -Wno-deprecated-gpu-targets 

rayTracer: main.cu *.h
	$(CC) main.cu $(CFLAGS) $(ARGS) -o $@

clean:
	@rm rayTracer ||:
