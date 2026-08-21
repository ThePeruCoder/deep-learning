# Compiler and Flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall

# Target Executable Name
TARGET = main

# Source Files
SRCS = src/main.cpp src/Perceptron.cpp src/Dataloader.cpp

# Object Files
OBJS = $(SRCS:.cpp=.o)

# Default Rule
all: $(TARGET)

# Linking rules 
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

# Compilation rule
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@	

# Clean rule
clean:
	rm -f $(OBJS) $(TARGET)