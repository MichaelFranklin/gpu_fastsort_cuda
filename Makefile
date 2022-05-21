#
# Makefile
# Author: Marius Rejdak
#

OBJ =  quickSort.out stlSort.out

all: $(OBJ)

GENCODE_SM35    := -gencode arch=compute_35,code=sm_35
GENCODE_SM50    := -gencode arch=compute_50,code=sm_50
GENCODE_SMXX    := -gencode arch=compute_50,code=compute_50
GENCODE_FLAGS   ?= $(GENCODE_SM35) $(GENCODE_SM50) $(GENCODE_SMXX)


%.out: %.cu utils.h cuda_utils.h
	nvcc $(GENCODE_FLAGS) -g -m64 $< -o $@

%.out: %.cpp utils.h
	g++ -O2 -m64 $< -o $@

clean:
	rm -f *.out *.csv

test:
	for file in $(OBJ); do \
		echo "Generate $$file"; \
    	bash -c "./$$file > $$file.csv"; \
	done
