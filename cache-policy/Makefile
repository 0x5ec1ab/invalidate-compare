NVCC = nvcc
FLAGS = -arch=sm_80 -Xptxas -O0

FILES = $(wildcard *.cu) 
BINS = $(FILES:.cu=)

all: $(BINS)

%: %.cu
	$(NVCC) $(FLAGS) $< -o $@

clean:
	rm -f $(BINS)

.PHONY: all clean

