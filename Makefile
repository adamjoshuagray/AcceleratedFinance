
cuda.o:
	nvcc --lib --relocatable-device-code true -o cuda.o CUDA/*.cu

test.o: cuda.o
	clang++ -Wall -Werror -c -o test.o Test/*.cpp

test: test.o
	nvcc *.o -lboost_unit_test_framework -o test

clean:
	nvcc --clean-targets --lib --relocatable-device-code true -o cuda.o CUDA/*.cu
	rm -f *.o
	rm -f test
