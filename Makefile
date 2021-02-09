CC=gcc
OPT=-O3
CFLAGS=-g -Wall -Wextra -Wno-unused-variable -Wshadow -Wno-sign-compare
LINKERS=-lm -pthread -fopenmp
OBJS=model.o layer.o utils.o matrix.o



predict_model: predict_model.c $(OBJS)
	$(CC) $(CFLAGS) $(OPT) -o $@ $^ $(LINKERS)

test_model: test_model.c $(OBJS)
	$(CC) $(CFLAGS) $(OPT) -o $@ $^ $(LINKERS)

train_model: train_model.c $(OBJS)
	$(CC) $(CFLAGS) $(OPT) -o $@ $^ $(LINKERS)

%.o: %.c %.h
	$(CC) $(CFLAGS) $(OPT) -c $< $(LINKERS)

clean:
	rm $(OBJS) main a.out train_model test_model extract

predict:
	make clean || true
	make predict_model
	/home/green-mint/dev/.venv/bin/python3 /home/green-mint/dev/c-nn/create_data.py
	./predict_model

matmul: matmul.c matrix.o
	$(CC) $(CFLAGS) -o $@ $^ $(LINKERS)

%: %.c
	$(CC) $(CFLAGS) -o $@ $^ $(LINKERS)