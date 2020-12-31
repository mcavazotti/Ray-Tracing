CC=nvcc
ARCH=sm_50
CFLAGS=-arch=$(ARCH) -Wno-deprecated-gpu-targets

rayTracer: main.cu *.h
	$(CC) main.cu $(CFLAGS) -o $@

clean:
	@rm rayTracer ||:
