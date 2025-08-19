# Compiler settings
CXX := g++

# Build type (Debug or Release)
BUILD_TYPE ?= Debug

ifeq ($(BUILD_TYPE), Debug)
    CXXFLAGS := -std=c++17 -g -O0 -DDEBUG -I./Tensor -I/opt/libtorch/include -I/opt/libtorch/include/torch/csrc/api/include
else
    CXXFLAGS := -std=c++17 -O2 -DNDEBUG -I./Tensor -I/opt/libtorch/include -I/opt/libtorch/include/torch/csrc/api/include
endif

LDFLAGS := -L/opt/libtorch/lib -Wl,-rpath,/opt/libtorch/lib
LIBS := -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda

# Auto-detect all .cpp files (recursively)
SRCS := $(shell find . -name '*.cpp')
OBJS := $(SRCS:.cpp=.o)
TARGET := main

# Main build rule
$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) $(OBJS) -o $@ $(LIBS)

# Compile each .cpp into .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Debug build
debug:
	$(MAKE) BUILD_TYPE=Debug

# Release build  
release:
	$(MAKE) BUILD_TYPE=Release

# Clean rule (removes all .o and the executable)
clean:
	rm -f $(OBJS) $(TARGET)

# Run the program
run: $(TARGET)
	./$(TARGET)

# Show compiler flags (useful for debugging build issues)
show-flags:
	@echo "CXXFLAGS: $(CXXFLAGS)"
	@echo "LDFLAGS: $(LDFLAGS)"
	@echo "LIBS: $(LIBS)"

.PHONY: clean run debug release show-flags