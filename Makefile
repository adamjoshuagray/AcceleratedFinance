cuda:
	mkdir -p Output
	nvcc --lib --relocatable-device-code true -o Output/cuda.o CUDA/*.cu

clean:
	rm -rf Output
