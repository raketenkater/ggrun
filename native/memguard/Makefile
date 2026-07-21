CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -Werror

.PHONY: all test clean

all: libggrun-memguard.so

libggrun-memguard.so: memguard.c
	$(CC) $(CFLAGS) -fPIC -shared -o $@ $< -ldl -pthread

libfakecuda.so: fakecuda.c
	$(CC) $(CFLAGS) -fPIC -shared -o $@ $<

test-target: test_target.c libfakecuda.so
	$(CC) $(CFLAGS) -o $@ test_target.c -L. -lfakecuda -Wl,-rpath,'$$ORIGIN'

test: libggrun-memguard.so test-target
	bash ./test.sh

clean:
	rm -f libggrun-memguard.so libfakecuda.so test-target
