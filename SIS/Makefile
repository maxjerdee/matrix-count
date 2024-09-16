CC = g++
CFLAGS = -std=c++17 -w -O2
LIBS =

# List of source files
SRCS = generate_samples.cpp preprocessing.cpp helpers.cpp estimates.cpp

# List of object files (one for each source file)
OBJS = $(SRCS:.cpp=.o)

# Main target
generate_samples: $(OBJS)
	$(CC) $(CFLAGS) -o generate_samples $(OBJS) $(LIBS)

# Rule to compile source files into object files
%.o: %.cpp globals.h preprocessing.h helpers.h generate_samples.h estimates.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f generate_samples $(OBJS)