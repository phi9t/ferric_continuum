# Hello World Example - C++ and Rust

## Week 1 Objectives

This week focuses on setting up a robust development environment and building foundational skills:

- Setting up Bazel and Spack for reproducible builds.
- Integrating C++ and Rust toolchains.
- Building a minimal "Hello World" example in both languages.
- Understanding hermetic builds, sandboxing, and toolchain configuration.
- Running basic tests to verify reproducibility.

This directory demonstrates Ferric Continuum's collocated C++ and Rust approach with a simple example that includes:

- Greeting generation
- Fibonacci sequence calculation
- Prime number checking

## Structure

Both C++ and Rust implementations live side-by-side:

```
ferric_continuum/hello/
├── BUILD.bazel           # Bazel build rules for both languages
├── hello.cc              # C++ main program
├── hello.rs              # Rust main program
├── hello_lib.hh          # C++ library header
├── hello_lib.cc          # C++ library implementation
├── hello_lib.rs          # Rust library implementation
├── hello_lib_test.cc     # C++ tests (using GoogleTest)
└── hello_lib_test.rs     # Rust tests (embedded in hello_lib.rs)
```

## Building

### Build C++ version
```bash
bazel build //ferric_continuum/hello:hello_cc
```

### Build Rust version
```bash
bazel build //ferric_continuum/hello:hello_rs
```

### Build both
```bash
bazel build //ferric_continuum/hello/...
```

## Running

### Run C++ program
```bash
bazel run //ferric_continuum/hello:hello_cc
```

### Run Rust program
```bash
bazel run //ferric_continuum/hello:hello_rs
```

## Testing

### Test C++ library
```bash
bazel test //ferric_continuum/hello:hello_lib_cc_test
```

### Test Rust library
```bash
bazel test //ferric_continuum/hello:hello_lib_rs_test
```

### Test both
```bash
bazel test //ferric_continuum/hello/...
```

## Comparing Implementations

Both implementations provide the same functionality:

### Greeting Generation
- **C++**: Uses `std::ostringstream` for string formatting
- **Rust**: Uses `format!` macro with cleaner syntax

### Fibonacci
- **C++**: Iterative approach with `uint64_t`
- **Rust**: Match expression with clear base cases, then iteration

### Prime Checking
- **C++**: Traditional loop with `std::sqrt`
- **Rust**: Similar logic but with functional style for `primes_up_to`

### Testing
- **C++**: External test file using GoogleTest framework
- **Rust**: Embedded tests using `#[cfg(test)]` module

## Learning Points

This example demonstrates:

1. **Side-by-side comparison** of C++ and Rust idioms
2. **Same algorithms** in different language styles
3. **Bazel integration** for multi-language builds
4. **Testing approaches** in both ecosystems
5. **Type systems** and safety guarantees

## Next Steps

After running this example, explore:
- Parallel implementations in `ferric_continuum/parallel/`
- Matrix operations in `ferric_continuum/matrix/`
- Distributed computing in `ferric_continuum/distributed/`
