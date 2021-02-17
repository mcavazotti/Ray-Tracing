CC=g++
CFLAGS=-fopenmp

rayTracer: main.cc *.h
	$(CC) main.cc $(CFLAGS) $(ARGS) -o $@

clean:
	@rm rayTracer ||:
