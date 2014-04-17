CXX = g++
OFLAGS = -g -o
CFLAGS = -shared	-fPIC	-o
SHARED_LIB_FLAG	=	`pkg-config --libs --cflags opencv`

all	:	train	detect	lib

detect	:	train	lib
	$(CXX)	$(OFLAGS)	detect	detect.cpp	./descriptor.so	$(SHARED_LIB_FLAG)

train	:	train.cpp	lib
	$(CXX)	$(OFLAGS)	train	train.cpp	./descriptor.so	$(SHARED_LIB_FLAG)	

lib	:	descriptor.cpp	descriptor.h	include.h	HOGtypes.h
	$(CXX)	$(CFLAGS)	descriptor.so	descriptor.cpp	$(SHARED_LIB_FLAG)


		

