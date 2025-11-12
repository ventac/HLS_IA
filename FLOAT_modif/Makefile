CC=gcc

IDIR = /usr/include/hdf5/serial/
CFLAGS = -I$(IDIR) -O3
LIBS = -lhdf5_serial -lm

# all: lenet_cnn_float lenet_cnn_fixed
all: lenet_cnn_fixed

# lenet_cnn_float: lenet_cnn_float.o fc.o 
# 	$(CC) -o lenet_cnn_float lenet_cnn_float.o fc.o  $(LIBS)
	
# lenet_cnn_float: lenet_cnn_float.o fc.o pool.o conv.o utils.o
# 	$(CC) -o lenet_cnn_float lenet_cnn_float.o fc.o pool.o conv.o utils.o $(LIBS)

lenet_cnn_fixed: lenet_cnn_fixed.o fc_fixed.o float_int.o 
	$(CC) -o lenet_cnn_fixed lenet_cnn_fixed.o fc_fixed.o float_int.o  $(LIBS)

# lenet_cnn_fixed: lenet_cnn_fixed.o fc_fixed.o float_int.o conv.o pool.o utils.o
# 	$(CC) -o lenet_cnn_fixed lenet_cnn_fixed.o fc_fixed.o float_int.o conv.o pool.o utils.o $(LIBS)


# lenet_cnn_float.o: lenet_cnn_float.c 
# 	$(CC) -c lenet_cnn_float.c $(CFLAGS)

lenet_cnn_fixed.o: lenet_cnn_fixed.c
	$(CC) -c lenet_cnn_fixed.c $(CFLAGS)

fc.o: fc.c 
	$(CC) -c fc.c $(CFLAGS)

fc_fixed.o: fc_fixed.c
	$(CC) -c fc_fixed.c $(CFLAGS)

float_int.o: float_int.c
	$(CC) -c float_int.c $(CFLAGS)

# pool.o: pool.c 
# 	$(CC) -c pool.c $(CFLAGS)

# conv.o: conv.c 
# 	$(CC) -c conv.c $(CFLAGS)

# utils.o: utils.c 
# 	$(CC) -c utils.c $(CFLAGS)
	
clean: 
	rm -f lenet_cnn_float lenet_cnn_fixed lenet_cnn_float.o lenet_cnn_fixed.o fc.o
# 	rm -f *.o lenet_cnn_float lenet_cnn_fixed
