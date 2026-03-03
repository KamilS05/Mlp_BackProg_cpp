# Minimal Makefile for the la1 project

CXX      := g++
CXXFLAGS := -std=c++11 -O2 -Wall -Wextra -I.

TARGET  := la1
BIN_DIR := bin

SRCS := la1.cpp imc/MultilayerPerceptron.cpp imc/util.cpp
OUT  := $(BIN_DIR)/$(TARGET)

.PHONY: all clean

all: $(OUT)

$(OUT): $(SRCS)
	mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(OUT)

clean:
	rm -rf $(BIN_DIR)
