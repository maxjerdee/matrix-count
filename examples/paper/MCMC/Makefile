CC = g++
CFLAGS = -std=c++17 -w -O2
LIBS =

# List of source files
SRCS = sample_binary_core.cpp preprocessing.cpp helpers.cpp estimates.cpp

# List of object files (one for each source file)
OBJS = $(SRCS:.cpp=.o)

# Main target
sample_binary_core: $(OBJS)
	$(CC) $(CFLAGS) -o sample_binary_core $(OBJS) $(LIBS)

# Rule to compile source files into object files
%.o: %.cpp globals.h preprocessing.h helpers.h sample_binary_core.h estimates.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f sample_binary_core $(OBJS)