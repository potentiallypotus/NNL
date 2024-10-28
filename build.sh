#!/bin/bash


# Compiler and flags
CXX="nvcc"
CUDAXX="nvcc"
CXXFLAGS="-std=c++17"

# Source and object files
SRC_DIR="src"
OBJ_DIR="obj"
FILES=("main.cpp" "NNL.cpp")
CUDA_FILES=("linear.cu")
TARGET="runNNL"

# Create object directory if it doesn't exist
mkdir -p $OBJ_DIR

compileFailed=0
# Function to compile a source file if necessary returns 0 if no files needed compilation
compile_if_needed() {
    local src_file="$1"
    local obj_file="$OBJ_DIR/${1%.cpp}.o"

    # Check if the object file exists or is older than the source file
    if [ ! -f "$obj_file" ] || [ "$src_file" -nt "$obj_file" ]; then
        echo "Compiling $src_file..."
        $CXX $CXXFLAGS -c -g -G -O0 "$src_file" -o "$obj_file"
	if [[ $? -eq 0 ]]; then
	    needsLinking=1
	else
	    compileFailed=1
	fi
	return 1
    else
        echo "$src_file is up to date, skipping compilation."
	return 0
    fi
}
compile_cuda_if_needed() {
    local src_file="$1"
    local obj_file="$OBJ_DIR/${1%.cu}.o"

    # Check if the object file exists or is older than the source file
    if [ ! -f "$obj_file" ] || [ "$src_file" -nt "$obj_file" ]; then
        echo "Compiling $src_file..."
        $CUDAXX $CXXFLAGS -c "$src_file" -o "$obj_file"
	needsLinking=1
	return 1
    else
        echo "$src_file is up to date, skipping compilation."
	return 0
    fi
}


# Compile each file if needed
needsLinking=0
for file in "${FILES[@]}"; do
	compile_if_needed "$file"
done
for file in "${CUDA_FILES[@]}"; do
	compile_cuda_if_needed "$file"
done


# Link the object files into the final executable
if [[ $needsLinking -eq 1 && $compileFailed -eq 0 ]]; then
	echo "Linking..."
	$CUDAXX $OBJ_DIR/*.o -o $TARGET
	echo "Build complete."
else
	echo "nothing to be done."
fi
