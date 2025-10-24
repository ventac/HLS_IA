# Compiler and flags
CC = gcc
IDIR = /usr/include/hdf5/serial/
CFLAGS = -I$(IDIR) -O3
LIBS = -lhdf5_serial -lm

# Targets
TARGET = lenet_cnn_fixed_example

# Source files
SRCS = lenet_cnn_fixed_example.c conv_fixed.c pool_fixed.c fc_fixed.c conversion_utils.c
OBJS = $(SRCS:.c=.o)

# Default target
all: $(TARGET)

# Link
$(TARGET): $(OBJS)
	$(CC) -o $@ $(OBJS) $(LIBS)

# Compile each .c into .o
%.o: %.c fixed_point.h
	$(CC) -c $< $(CFLAGS)

# Clean
clean:
	rm -f $(OBJS) $(TARGET)