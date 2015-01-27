
CC = clang++

all: tests.cpp; $(CC) -o tests.tst tests.cpp -I /usr/local/include -L/usr/local/lib -std=c++11 -larmadillo

runtest: all; ./tests.tst

clean:; rm tests.tst
