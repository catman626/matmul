# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/catman/ta/matmul

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/catman/ta/matmul/build

# Include any dependencies generated for this target.
include CMakeFiles/ascendc_kernels_cpu.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/ascendc_kernels_cpu.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/ascendc_kernels_cpu.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ascendc_kernels_cpu.dir/flags.make

CMakeFiles/ascendc_kernels_cpu.dir/matmul_custom.cpp.o: CMakeFiles/ascendc_kernels_cpu.dir/flags.make
CMakeFiles/ascendc_kernels_cpu.dir/matmul_custom.cpp.o: ../matmul_custom.cpp
CMakeFiles/ascendc_kernels_cpu.dir/matmul_custom.cpp.o: CMakeFiles/ascendc_kernels_cpu.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/catman/ta/matmul/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ascendc_kernels_cpu.dir/matmul_custom.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ascendc_kernels_cpu.dir/matmul_custom.cpp.o -MF CMakeFiles/ascendc_kernels_cpu.dir/matmul_custom.cpp.o.d -o CMakeFiles/ascendc_kernels_cpu.dir/matmul_custom.cpp.o -c /home/catman/ta/matmul/matmul_custom.cpp

CMakeFiles/ascendc_kernels_cpu.dir/matmul_custom.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ascendc_kernels_cpu.dir/matmul_custom.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/catman/ta/matmul/matmul_custom.cpp > CMakeFiles/ascendc_kernels_cpu.dir/matmul_custom.cpp.i

CMakeFiles/ascendc_kernels_cpu.dir/matmul_custom.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ascendc_kernels_cpu.dir/matmul_custom.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/catman/ta/matmul/matmul_custom.cpp -o CMakeFiles/ascendc_kernels_cpu.dir/matmul_custom.cpp.s

# Object files for target ascendc_kernels_cpu
ascendc_kernels_cpu_OBJECTS = \
"CMakeFiles/ascendc_kernels_cpu.dir/matmul_custom.cpp.o"

# External object files for target ascendc_kernels_cpu
ascendc_kernels_cpu_EXTERNAL_OBJECTS =

libascendc_kernels_cpu.so: CMakeFiles/ascendc_kernels_cpu.dir/matmul_custom.cpp.o
libascendc_kernels_cpu.so: CMakeFiles/ascendc_kernels_cpu.dir/build.make
libascendc_kernels_cpu.so: /home/catman/Ascend/ascend-toolkit/latest/tools/tikicpulib/lib/libtikicpulib_cceprint.so
libascendc_kernels_cpu.so: /home/catman/Ascend/ascend-toolkit/latest/tools/tikicpulib/lib/libtikicpulib_npuchk.so
libascendc_kernels_cpu.so: /home/catman/Ascend/ascend-toolkit/latest/tools/tikicpulib/lib/libtikicpulib_stubreg.so
libascendc_kernels_cpu.so: CMakeFiles/ascendc_kernels_cpu.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/catman/ta/matmul/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libascendc_kernels_cpu.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ascendc_kernels_cpu.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ascendc_kernels_cpu.dir/build: libascendc_kernels_cpu.so
.PHONY : CMakeFiles/ascendc_kernels_cpu.dir/build

CMakeFiles/ascendc_kernels_cpu.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ascendc_kernels_cpu.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ascendc_kernels_cpu.dir/clean

CMakeFiles/ascendc_kernels_cpu.dir/depend:
	cd /home/catman/ta/matmul/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/catman/ta/matmul /home/catman/ta/matmul /home/catman/ta/matmul/build /home/catman/ta/matmul/build /home/catman/ta/matmul/build/CMakeFiles/ascendc_kernels_cpu.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ascendc_kernels_cpu.dir/depend
