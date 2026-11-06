CC=gcc
CFLAGS=-O2 -std=c11 -Wall -Wextra

chv: chv.c
	$(CC) $(CFLAGS) -o chv chv.c

test: chv
	./chv --selftest

clean:
	rm -f chv
