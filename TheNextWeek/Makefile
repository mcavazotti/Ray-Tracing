CC=g++
CFLAGS=-fopenmp -O3

rayTracer: main.cc *.h
	$(CC) main.cc $(CFLAGS) -o $@

clean:
	@rm rayTracer ||:
