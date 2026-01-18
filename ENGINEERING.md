# ENGINEERING.md

Engineering standards and best practices for Ferric Continuum development.
For a focused C++ fundamentals guide, see `CXX_ENGINEERING.md`.

---

## Table of Contents

1. [Philosophy](#philosophy)
2. [C++ Fundamentals](#c-fundamentals)
3. [Project-Specific Standards](#project-specific-standards)
   - [Namespace Usage](#namespace-usage-c)
   - [Logging Standards](#logging-standards)
   - [Testing and Benchmarking Standards](#testing-and-benchmarking-standards)
4. [Rust Best Practices](#rust-best-practices)
5. [Tooling and Automation](#tooling-and-automation)
6. [Recommendations for This Project](#recommendations-for-this-project)

---

## Philosophy

Ferric Continuum prioritizes **correctness, performance, and maintainability** in that order. Code should be:

- **Self-documenting** - express intent directly through types and names
- **Safe by default** - prevent errors at compile time when possible
- **Observable** - instrumented for profiling and debugging
- **Reproducible** - hermetic builds with deterministic outputs

---

## C++ Fundamentals

For C++ style, ownership, class design, logging, testing, and the quick checklist,
see `CXX_ENGINEERING.md`.

---

## Project-Specific Standards

These standards are specific to Ferric Continuum and complement the general C++ and Rust guidelines.

### Namespace Usage (C++)

See `CXX_ENGINEERING.md` for the full namespace usage rules and examples.

### Logging Standards

**Use structured logging instead of raw print statements** for all code beyond trivial examples.

#### C++ Logging

Use Abseil logging for C++ code. See `CXX_ENGINEERING.md` for setup, usage, and examples.

#### Rust Logging

Use the **tracing** crate for all Rust code, including demos and examples.

**Macros:** `trace!`, `debug!`, `info!`, `warn!`, `error!`

**Initialization** (required in `main()`):
```rust
use tracing::{info, Level};
use tracing_subscriber;

fn main() {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();
    
    info!("Program started");
    // ... rest of program ...
}
```

**BUILD.bazel dependencies:**
```python
rust_binary(
    name = "my_program",
    srcs = ["my_program.rs"],
    edition = "2021",
    deps = [
        "@crates//:tracing",
        "@crates//:tracing-subscriber",
        # ... other deps
    ],
)
```

**Usage examples:**
```rust
use tracing::{info, warn, error, debug};

// ❌ BAD - Raw output
println!("Processing {} items", count);
eprintln!("Error: Buffer full");

// ✅ GOOD - Structured logging
info!(count, "Processing items");
warn!(usage = %usage, "Buffer nearly full");
error!(size, "Failed to allocate memory");

// With structured fields
info!(
    rows = matrix.rows,
    cols = matrix.cols,
    duration_ms = elapsed.as_millis(),
    "Matrix multiplication complete"
);

// Spans for tracing execution
use tracing::instrument;

#[instrument]
async fn process_data(data: &Data) -> Result<Output> {
    debug!("Starting data processing");
    // Function execution is automatically traced
    // ...
}
```

#### Benefits of Structured Logging

- **Configurable**: Output levels and destinations can be changed at runtime
- **Structured data**: Machine-parseable for log aggregation and analysis
- **Performance**: Can be compiled out or filtered with minimal overhead
- **Integration**: Works with observability tools (Prometheus, Grafana, ELK stack)
- **Context**: Automatically captures file, line, and module information
- **Distributed tracing**: Spans enable tracking across async operations and microservices

#### When Raw Output Is Acceptable

- **Build scripts**: Simple build/generation scripts that aren't part of the runtime system
- **Quick debugging**: Temporary debugging during development (remove before commit)
- **Interactive tools**: CLI tools where user-facing output is the primary purpose (though consider `clap` with logging)

**All demo programs and examples use structured logging** to demonstrate best practices. This includes files in `ferric_continuum/*/` directories with `*_demo.{cc,rs}` suffixes. Production code must always use the appropriate logging framework.

### Testing and Benchmarking Standards

#### Unit Testing

**C++ Testing (Google Test)**
See `CXX_ENGINEERING.md` for C++ testing examples and conventions.

**Rust Testing (Built-in)**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_functionality() {
        let obj = MyStruct::new();
        assert_eq!(obj.compute(5), 25);
        assert!(obj.is_valid());
    }

    #[test]
    #[should_panic(expected = "invalid argument")]
    fn test_edge_cases() {
        let obj = MyStruct::new();
        obj.compute(-1);
    }
}
```

#### Benchmarking

**C++ Benchmarks (Google Benchmark)**
See `CXX_ENGINEERING.md` for C++ benchmarking examples and conventions.

**Rust Benchmarks (Criterion)**

```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn matrix_multiply_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("MatrixMultiply");

    for size in [8, 16, 32, 64, 128, 256, 512] {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &size,
            |b, &size| {
                let a = Matrix::new(size, size);
                let b = Matrix::new(size, size);
                b.iter(|| &a * &b);
            }
        );
    }

    group.finish();
}

criterion_group!(benches, matrix_multiply_benchmark);
criterion_main!(benches);
```

#### Performance Profiling

**Tools by Platform:**

- **C++**: Google Benchmark, perf, valgrind, gprof
- **Rust**: Criterion.rs, cargo flamegraph, perf
- **Cross-platform**: Intel VTune, AMD μProf

**Running Profilers:**

```bash
# C++ with perf
bazel build --config=opt //cpp:matrix_mul_benchmark
perf record -g bazel-bin/cpp/matrix_mul_benchmark
perf report

# Rust with flamegraph
cargo install flamegraph
cargo flamegraph --bench my_benchmark

# Valgrind for memory profiling
valgrind --tool=massif bazel-bin/cpp/my_binary
ms_print massif.out.*
```

**Best Practices:**
- Always profile in release/optimized builds
- Run benchmarks multiple times to account for variance
- Track performance over time with baseline comparisons
- Document performance characteristics in code comments
- Commit benchmark results to track regressions (via PerfSmith agent)

---

## Rust Best Practices

For Rust components in Ferric Continuum, follow these principles:

### Ownership & Borrowing
- Let the borrow checker guide your design
- Prefer borrowing (`&T`, `&mut T`) over ownership (`T`)
- Use `Rc`/`Arc` only when shared ownership is truly needed
- Avoid `RefCell`/`Mutex` unless interior mutability is required

### Error Handling
- Use `Result<T, E>` for recoverable errors
- Use `Option<T>` for optional values
- Propagate errors with `?` operator
- Reserve `panic!` for unrecoverable errors and bugs

```rust
// Good: composable error handling
fn read_config(path: &Path) -> Result<Config, io::Error> {
    let contents = fs::read_to_string(path)?;
    let config = parse_config(&contents)?;
    Ok(config)
}
```

### Type Safety
- Use newtype pattern for domain concepts
- Leverage `enum` for state machines and variants
- Make invalid states unrepresentable

```rust
// Good: invalid states impossible
struct UserId(u64);
struct UserName(String);

struct User {
    id: UserId,
    name: UserName,
}
```

### Performance
- Avoid allocations in hot paths
- Use iterators and `collect()` instead of manual loops
- Profile with `cargo bench` and `criterion`
- Use `#[inline]` judiciously for small, hot functions

### Async Code (Tokio)
- Prefer async/await over manual futures
- Avoid blocking operations in async contexts
- Use `tokio::spawn` for concurrent tasks
- Be careful with `Arc<Mutex<T>>` - consider message passing

---

## Tooling and Automation

Automated tooling ensures code quality, consistency, and reduces manual effort. Ferric Continuum uses industry-standard tools for formatting and linting.

### C++ Tools

#### clang-format

**Purpose**: Automatic code formatting for C++ files.

**Configuration**: `.clang-format` in repository root (based on Google style).

**Usage**:
```bash
# Format a single file
clang-format -i path/to/file.cpp

# Format all C++ files in project
find . -name "*.cpp" -o -name "*.h" -o -name "*.cc" -o -name "*.hpp" | xargs clang-format -i

# Check formatting without modifying (CI mode)
clang-format --dry-run --Werror path/to/file.cpp

# Use the provided script
./scripts/format.sh          # Format all files
./scripts/format.sh --check  # Check only (CI)
```

**Key Benefits**:
- Eliminates formatting debates
- Ensures consistency across the codebase
- Automatic fixes for most style issues

#### clangd

**Purpose**: C++ Language Server Protocol (LSP) implementation for IDE/editor integration.

**Features**:
- Code completion
- Go-to-definition
- Find references
- Real-time diagnostics
- Inline documentation

**Configuration**: `.clangd` in repository root.

**Setup**:
```bash
# Generate compile_commands.json for clangd
bazel run @hedron_compile_commands//:refresh_all

# Most editors auto-detect clangd; verify it's in your PATH
which clangd
```

**Editor Support**:
- **VS Code / Cursor / Trae**: Install "clangd" extension from marketplace
- **Emacs / Doom Emacs**: Use lsp-mode with clangd (see Editor Configuration section below)

#### clang-tidy

**Purpose**: C++ static analyzer and linter with auto-fix capabilities.

**Configuration**: `.clang-tidy` in repository root.

**Usage**:
```bash
# Run clang-tidy on a single file
clang-tidy path/to/file.cpp -- -I./include

# Run with auto-fix
clang-tidy --fix path/to/file.cpp -- -I./include

# Run on all files with compile_commands.json
clang-tidy -p . path/to/file.cpp

# Use the provided script
./scripts/lint.sh           # Lint and report issues
./scripts/lint.sh --fix     # Lint and auto-fix
```

**Enabled Checks** (see `.clang-tidy` for full list):
- `modernize-*` - Suggests modern C++ idioms (auto, nullptr, range-for)
- `readability-*` - Improves code readability
- `performance-*` - Identifies performance issues
- `bugprone-*` - Catches common bugs
- `cppcoreguidelines-*` - Enforces C++ Core Guidelines

**Auto-fixable Issues**:
- Use `nullptr` instead of `NULL`
- Use `auto` for obvious types
- Use range-based for loops
- Add `override` keywords
- Remove unnecessary copies

### Rust Tools

#### rustfmt

**Purpose**: Official Rust code formatter.

**Configuration**: `rustfmt.toml` in repository root.

**Usage**:
```bash
# Format all Rust code in project
cargo fmt

# Check formatting without modifying
cargo fmt -- --check

# Format specific files
rustfmt src/main.rs
```

**Integration**:
- Runs automatically in most Rust-aware editors
- Required in CI pipeline

#### clippy

**Purpose**: Official Rust linter with hundreds of lint rules and auto-fix support.

**Configuration**: `.clippy.toml` and `Cargo.toml` lint settings.

**Usage**:
```bash
# Run clippy on all targets
cargo clippy --all-targets --all-features

# Run with auto-fix
cargo clippy --fix --all-targets --all-features

# Deny warnings (CI mode)
cargo clippy -- -D warnings

# Explain a specific lint
cargo clippy -- -W clippy::lint-name --explain
```

**Key Lint Categories**:
- Correctness (deny by default) - Probable bugs
- Suspicious - Code that's likely wrong
- Style - Code style recommendations
- Complexity - Overly complex code patterns
- Performance - Performance issues
- Pedantic - Extra nitpicky lints (opt-in)

**Common Auto-fixable Issues**:
- Unnecessary clones
- Redundant closures
- Verbose iterator patterns
- Inefficient string operations

### Integrated Automation Scripts

#### scripts/format.sh

Formats all C++ and Rust code in the repository.

```bash
# Format all code
./scripts/format.sh

# Check formatting only (exits with error if formatting needed)
./scripts/format.sh --check

# Format specific language
./scripts/format.sh --cpp-only
./scripts/format.sh --rust-only
```

#### scripts/lint.sh

Runs all linters with optional auto-fix.

```bash
# Run all linters (report mode)
./scripts/lint.sh

# Run with auto-fix
./scripts/lint.sh --fix

# Run specific linter
./scripts/lint.sh --cpp-only
./scripts/lint.sh --rust-only
```

### Bazel Integration

The `.bazelrc` file includes:

```bash
# Build with all warnings
build --copt=-Wall --copt=-Wextra

# Run clang-tidy on C++ builds
build --aspects=@bazel_clang_tidy//clang_tidy:clang_tidy.bzl%clang_tidy_aspect
build --output_groups=report

# Format check target
bazel run //:format_check  # Check formatting
bazel run //:format        # Apply formatting
```

### Pre-commit Integration (Optional)

For automatic formatting/linting before commits:

```bash
# Install pre-commit hook
./scripts/install-hooks.sh

# Manually run pre-commit checks
./scripts/pre-commit.sh
```

**Hook Actions**:
1. Run clang-format on staged C++ files
2. Run rustfmt on staged Rust files
3. Run clang-tidy on modified C++ files
4. Run clippy on Rust workspace
5. Fail commit if issues found (or auto-fix if configured)

### CI/CD Integration

All formatting and linting tools run in continuous integration:

```yaml
# CI Pipeline Steps
- name: Check Formatting
  run: ./scripts/format.sh --check

- name: Run Linters
  run: ./scripts/lint.sh

- name: Run Clippy
  run: cargo clippy --all-targets -- -D warnings
```

### Tool Installation

**C++ Tools**:
```bash
# Ubuntu/Debian
sudo apt install clang-format clang-tidy clangd

# macOS
brew install clang-format llvm

# From source or LLVM releases
# https://releases.llvm.org/
```

**Rust Tools**:
```bash
# Install rustfmt and clippy (included with rustup)
rustup component add rustfmt clippy

# Update to latest versions
rustup update
```

### Best Practices

1. **Format before committing**: Run `./scripts/format.sh` before every commit
2. **Fix lint issues**: Address linter warnings; don't suppress unless justified
3. **Use auto-fix judiciously**: Review auto-fixes before committing
4. **Configure your editor**: Enable format-on-save for both C++ and Rust
5. **Keep tools updated**: Update clang-tools and Rust components regularly
6. **Document suppressions**: If suppressing a lint, add a comment explaining why

### Quick Reference: Daily Workflow

```bash
# 1. Before starting work - ensure tools are available
which clang-format clang-tidy clangd
cargo fmt --version
cargo clippy --version

# 2. During development - format frequently
./scripts/format.sh                    # Format all code
./scripts/format.sh --cpp-only         # C++ only
./scripts/format.sh --rust-only        # Rust only

# 3. Before committing - check and fix issues
./scripts/format.sh --check            # Verify formatting
./scripts/lint.sh                      # Check for linting issues
./scripts/lint.sh --fix                # Auto-fix what's possible

# 4. Review changes
git diff                               # Review all changes including auto-fixes

# 5. Bazel integration (optional)
bazel build --config=clippy //...      # Rust linting
bazel build --config=rustfmt //...     # Rust formatting check
bazel build --config=ci //...          # Full CI checks locally
```

### CI/CD Pipeline Integration

```yaml
# Example CI pipeline configuration
name: Code Quality

on: [push, pull_request]

jobs:
  format-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install C++ tools
        run: sudo apt install clang-format clang-tidy

      - name: Install Rust tools
        run: rustup component add rustfmt clippy

      - name: Check formatting
        run: ./scripts/format.sh --check

      - name: Run linters
        run: ./scripts/lint.sh

      - name: Build with CI config
        run: bazel build --config=ci //...
```

### Common Auto-Fix Examples

#### C++ Auto-Fixes (clang-tidy)

```cpp
// Before: Using NULL instead of nullptr
void* ptr = NULL;

// After: Modern C++ (auto-fixed by clang-tidy)
void* ptr = nullptr;
```

```cpp
// Before: Missing override keyword
class Base {
  virtual void foo() {}
};

class Derived : public Base {
  void foo() {}  // Warning: missing override
};

// After: Auto-fixed
class Derived : public Base {
  void foo() override {}
};
```

```cpp
// Before: Unnecessary copy
for (std::string item : vec) {  // Copies each string
  process(item);
}

// After: Auto-fixed to use const reference
for (const std::string& item : vec) {
  process(item);
}
```

```cpp
// Before: Old-style loop
for (size_t i = 0; i < vec.size(); ++i) {
  process(vec[i]);
}

// After: Modernized (auto-fixed)
for (auto& item : vec) {
  process(item);
}
```

#### Rust Auto-Fixes (clippy)

```rust
// Before: Unnecessary clone
let s = String::from("hello");
let t = s.clone();
println!("{}", s);

// After: Auto-fixed - remove unnecessary clone
let s = String::from("hello");
let t = &s;
println!("{}", s);
```

```rust
// Before: Inefficient string operation
let s = format!("Hello");

// After: Auto-fixed
let s = "Hello".to_string();
```

```rust
// Before: Redundant closure
let doubled: Vec<i32> = vec.iter().map(|x| x * 2).collect();

// After: Auto-fixed (when appropriate)
let doubled: Vec<i32> = vec.iter().map(|x| x * 2).collect();
// Note: clippy suggests improvements based on context
```

### Troubleshooting

**Issue**: `clang-format: command not found`
```bash
# Ubuntu/Debian
sudo apt install clang-format

# macOS
brew install clang-format

# Verify installation
clang-format --version
```

**Issue**: `clang-tidy` requires `compile_commands.json`
```bash
# Generate compilation database with Bazel
bazel run @hedron_compile_commands//:refresh_all

# Or use bear for non-Bazel projects
bear -- make
```

**Issue**: Rust tools not found
```bash
# Install rustup if needed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Add components
rustup component add rustfmt clippy

# Update to latest
rustup update
```

**Issue**: Format/lint scripts fail on CI
```bash
# Ensure scripts are executable
chmod +x scripts/format.sh scripts/lint.sh

# Test locally first
./scripts/format.sh --check
./scripts/lint.sh
```

**Issue**: Too many false positives from clang-tidy
```bash
# Disable specific checks in .clang-tidy
# Or suppress inline with NOLINT comment
void legacy_function() {  // NOLINT(modernize-use-trailing-return-type)
  // Justification: Legacy API compatibility
}
```

### Performance Tips

**For large codebases:**
```bash
# Run formatters in parallel on specific directories
find cpp/ -name "*.cpp" | xargs -P 4 clang-format -i

# Lint only changed files
git diff --name-only --diff-filter=ACM | grep -E '\.(cpp|cc|h|hpp)$' | xargs clang-tidy --fix

# Cache clang-tidy results
export CLANG_TIDY_CACHE_DIR=~/.cache/clang-tidy
```

**For faster feedback during development:**
```bash
# Format only staged files
git diff --cached --name-only --diff-filter=ACM | grep -E '\.(cpp|cc|h|hpp)$' | xargs clang-format -i

# Lint only current directory
./scripts/lint.sh --cpp-only $(pwd)
```

### Editor Configuration Examples

#### VS Code / Cursor / Trae

Create or update `.vscode/settings.json` in your project root:

```json
{
  "editor.formatOnSave": true,
  "editor.defaultFormatter": null,

  // C++ Configuration
  "C_Cpp.codeAnalysis.clangTidy.enabled": true,
  "C_Cpp.codeAnalysis.clangTidy.path": "/usr/bin/clang-tidy",
  "C_Cpp.default.configurationProvider": "llvm-vs-code-extensions.vscode-clangd",

  // clangd LSP
  "clangd.path": "/usr/bin/clangd",
  "clangd.arguments": [
    "--background-index",
    "--clang-tidy",
    "--completion-style=detailed",
    "--header-insertion=iwyu",
    "--pch-storage=memory"
  ],

  // Rust Configuration
  "[rust]": {
    "editor.defaultFormatter": "rust-lang.rust-analyzer",
    "editor.formatOnSave": true
  },
  "rust-analyzer.check.command": "clippy",
  "rust-analyzer.checkOnSave": true,
  "rust-analyzer.rustfmt.extraArgs": [
    "+nightly"
  ],

  // File associations
  "files.associations": {
    "*.bazel": "starlark",
    "BUILD": "starlark",
    "WORKSPACE": "starlark"
  }
}
```

**Required Extensions:**
- **C++**: [clangd](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-clangd)
- **Rust**: [rust-analyzer](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer)
- **Bazel**: [Bazel](https://marketplace.visualstudio.com/items?itemName=BazelBuild.vscode-bazel)

**Install Extensions:**
```bash
code --install-extension llvm-vs-code-extensions.vscode-clangd
code --install-extension rust-lang.rust-analyzer
code --install-extension BazelBuild.vscode-bazel
```

#### Emacs (Doom Emacs)

Add to your Doom Emacs configuration (`~/.doom.d/config.el`):

```elisp
;; C++ Configuration with clangd and clang-format
(after! lsp-mode
  (setq lsp-clients-clangd-args
        '("--background-index"
          "--clang-tidy"
          "--completion-style=detailed"
          "--header-insertion=iwyu"
          "--pch-storage=memory"))

  ;; Enable clangd for C/C++
  (setq lsp-clients-clangd-executable "/usr/bin/clangd"))

;; Auto-format on save for C/C++
(add-hook 'c-mode-hook
          (lambda ()
            (add-hook 'before-save-hook #'clang-format-buffer nil 'local)))

(add-hook 'c++-mode-hook
          (lambda ()
            (add-hook 'before-save-hook #'clang-format-buffer nil 'local)))

;; Rust Configuration with rust-analyzer
(after! rustic
  (setq rustic-format-on-save t)
  (setq rustic-lsp-server 'rust-analyzer)
  (setq rustic-analyzer-command '("rust-analyzer"))

  ;; Use clippy for checks
  (setq rustic-flycheck-clippy-params "--all-targets --all-features"))

;; Bazel support
(use-package! bazel
  :mode (("\\.bazel\\'" . bazel-mode)
         ("BUILD\\'" . bazel-mode)
         ("WORKSPACE\\'" . bazel-mode)))
```

**Required Doom modules** (add to `~/.doom.d/init.el`):

```elisp
(doom! :lang
       (cc +lsp)           ; C/C++ with LSP
       (rust +lsp)         ; Rust with LSP

       :tools
       lsp                 ; Language Server Protocol
       format              ; Auto-formatting

       :checkers
       syntax)             ; Syntax checking
```

**Install required packages:**
```bash
# Ensure clang-format package is available
doom sync
doom install

# For clang-format buffer command
M-x package-install RET clang-format RET
```

**Alternative: using format-all (simpler)**

```elisp
;; In config.el
(use-package! format-all
  :commands format-all-mode
  :hook (prog-mode . format-all-mode)
  :config
  (setq-default format-all-formatters
                '(("C" (clang-format))
                  ("C++" (clang-format))
                  ("Rust" (rustfmt)))))
```

---

## Recommendations for This Project

Given Ferric Continuum's goals (performance, HPC, distributed computing), we recommend:

### C++ Components
1. **Follow C++ Core Guidelines as primary reference**
   - Allow exceptions (unlike Google style) for error handling
   - Use RTTI sparingly, but don't ban it entirely
   - Target C++20, prepare for C++23 features

2. **Google Style Guide for code organization**
   - Namespace conventions
   - Header file practices
   - Naming conventions

3. **Performance-critical code**
   - Profile before optimizing
   - Use compiler-specific attributes when beneficial
   - Document performance assumptions and constraints

### Rust Components
1. **Standard Rust idioms**
   - Follow Clippy recommendations
   - Use `rustfmt` for formatting
   - Run `cargo clippy` in CI

2. **Async runtime (Tokio)**
   - Use for I/O-bound operations (network, file I/O)
   - Avoid for CPU-bound work (use thread pools instead)

3. **FFI with C++**
   - Use `cxx` crate for safe C++/Rust interop
   - Minimize boundary crossings
   - Document ownership transfer clearly

### Testing & Quality
- Unit tests for all components
- Integration tests for agent interactions
- Benchmark suite for performance tracking
- Use sanitizers (ASan, TSan, UBSan) in CI
- Document performance characteristics

### Documentation
- Document design decisions in code comments
- Use Doxygen for C++ API docs
- Use rustdoc for Rust API docs
- Maintain architectural decision records (ADRs)

---

## References

- [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)
- [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- [Effective Modern C++](https://www.oreilly.com/library/view/effective-modern-c/9781491908419/) by Scott Meyers
- [The Rust Programming Language](https://doc.rust-lang.org/book/)
