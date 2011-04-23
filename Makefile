CC=gcc

all: tests

bcutil.o: bcutil.h

tests = bctest_1d bctest_2d bctest_mmap

tests : $(tests)

$(tests) : % : %.o bcutil.o
	$(CC) -o $@ $^
clean:
	rm $(tests) *.o