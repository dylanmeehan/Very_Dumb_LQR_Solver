all:
	g++ -I ~/eigen/ main.cpp -o lqr_test

clean:
	rm -f lqr_test
	rm -f *.o
