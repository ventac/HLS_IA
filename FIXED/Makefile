CC = gcc
IDIR = /usr/include/hdf5/serial/
CFLAGS = -I$(IDIR) -O3
LIBS = -lhdf5_serial -lm

TARGET = lenet_cnn_fixed

SRCS = lenet_cnn_fixed.c \
       conv_fixed.c \
       pool_fixed.c \
       fc_fixed.c \
       conversion_utils.c \
       utils.c \
       globals.c

OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) -o $@ $(OBJS) $(LIBS)

%.o: %.c
	$(CC) -c $< $(CFLAGS)

clean:
	rm -f $(OBJS) $(TARGET)