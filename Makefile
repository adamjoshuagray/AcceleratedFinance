cuda:
	mkdir -p Output
	nvcc --lib --relocatable-device-code true -o Output/cuda.o CUDA/*.cu

clean:
	nvcc --clean-targets --lib --relocatable-device-code true -o Output/cuda.o CUDA/*.cu
	rm -rf Output
