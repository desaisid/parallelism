ARCH = sm_20

%.o:%.cu
	nvcc -c -arch ${ARCH} $< -o $@
	
%_cuda:%.o
	gcc -locelot $< -o $@ 
