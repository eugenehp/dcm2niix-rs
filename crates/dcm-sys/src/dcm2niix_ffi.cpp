// Rust entry: upstream main_console.cpp renamed so we can expose a C ABI.

#define main dcm2niix_cpp_main
#include "main_console.cpp"
#undef main

extern "C" int dcm2niix_run(int argc, char **argv) {
	return dcm2niix_cpp_main(argc, const_cast<const char **>(argv));
}
