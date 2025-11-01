# Contributing to Ferric Continuum

Thank you for your interest in contributing to Ferric Continuum! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Requirements](#testing-requirements)
- [Submitting Changes](#submitting-changes)
- [Documentation](#documentation)

## Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## Getting Started

### Prerequisites

- **Bazel 8.4.2+** - Build system
- **Python 3.12+** - For agents and tooling
- **Rust 1.81.0+** - Managed by Bazel
- **C++ compiler** - Clang 18+ or GCC 13+
- **Git** - Version control

### Setting Up Your Development Environment

1. **Fork the repository** on GitHub

2. **Clone your fork:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/ferric_continuum.git
   cd ferric_continuum
   ```

3. **Add upstream remote:**
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/ferric_continuum.git
   ```

4. **Verify installation:**
   ```bash
   bazel --version
   python3 --version
   ```

5. **Build and test:**
   ```bash
   bazel build //...
   bazel test //...
   ```

## Development Workflow

### Creating a Branch

Create a descriptive branch name:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

Branch naming conventions:
- `feature/*` - New features
- `fix/*` - Bug fixes
- `docs/*` - Documentation updates
- `refactor/*` - Code refactoring
- `test/*` - Test additions/improvements

### Making Changes

1. **Make your changes** following our [coding standards](#coding-standards)
2. **Add tests** for new functionality
3. **Update documentation** as needed
4. **Format your code:**
   ```bash
   # Format C++ code
   ./scripts/format.sh
   
   # Or manually
   clang-format -i path/to/file.cc
   rustfmt path/to/file.rs
   ```

5. **Run linters:**
   ```bash
   ./scripts/lint.sh
   ```

6. **Run tests:**
   ```bash
   bazel test //...
   ```

### Committing Changes

Write clear, descriptive commit messages:

```
Short (50 chars or less) summary

More detailed explanatory text, if necessary. Wrap it to about 72
characters. The blank line separating the summary from the body is
critical (unless you omit the body entirely).

- Bullet points are okay
- Use present tense: "Add feature" not "Added feature"
- Reference issues: Fixes #123
```

## Coding Standards

We follow strict coding standards documented in [ENGINEERING.md](ENGINEERING.md). Key points:

### C++ Standards

- **Style Guide**: Google C++ Style Guide and C++ Core Guidelines
- **Header files**: Use `.hh` extension
- **Source files**: Use `.cc` extension
- **Include guards**: Use `#pragma once`
- **Namespace usage**: Never use `using namespace` - use fully qualified names or namespace aliases
- **Logging**: Use Abseil logging (`LOG(INFO)`, etc.)
- **Testing**: Google Test for unit tests
- **Formatting**: `clang-format` with project config

Example:
```cpp
#pragma once

#include "absl/log/log.h"

namespace ferric::foundation {

// Use namespace alias for brevity
namespace ff = ferric::foundation;

class MyClass {
 public:
  void do_something();
};

}  // namespace ferric::foundation
```

### Rust Standards

- **Edition**: Rust 2021
- **Formatting**: `rustfmt` with project config
- **Linting**: `clippy` with project config
- **Logging**: `tracing` crate with structured logging
- **Testing**: Built-in test framework
- **Error handling**: `anyhow` or `thiserror` for errors

Example:
```rust
use tracing::{info, Level};

pub fn my_function() -> Result<(), anyhow::Error> {
    info!("Starting operation");
    // ... implementation
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_my_function() {
        assert!(my_function().is_ok());
    }
}
```

### General Standards

- **Line length**: 100 characters (C++ and Rust)
- **Indentation**: 2 spaces (no tabs)
- **Comments**: Clear, concise, explaining *why* not *what*
- **Documentation**: Every public API must be documented
- **Tests**: Every function must have unit tests

## Testing Requirements

### Unit Tests

Every new function must have comprehensive unit tests:

```cpp
// C++ - Google Test
TEST(MyClassTest, DescriptiveTestName) {
  MyClass obj;
  EXPECT_EQ(obj.method(), expected_value);
  EXPECT_TRUE(obj.is_valid());
}
```

```rust
// Rust - Built-in test framework
#[test]
fn test_my_function() {
    let result = my_function();
    assert_eq!(result, expected_value);
}
```

### Test Coverage Requirements

- **New features**: 100% coverage of new code
- **Bug fixes**: Test that reproduces the bug + verification
- **Refactoring**: Existing tests must pass
- **Edge cases**: Test boundary conditions, errors, null/None values

### Running Tests

```bash
# Run all tests
bazel test //...

# Run specific test
bazel test //ferric_continuum/foundation:value_semantics_test_cc

# Run with verbose output
bazel test //... --test_output=all

# Run only C++ tests
bazel test //... --test_tag_filters=cpp

# Run only Rust tests
bazel test //... --test_tag_filters=rust
```

## Submitting Changes

### Before Submitting

Checklist before creating a pull request:

- [ ] Code follows project style guidelines
- [ ] All tests pass locally
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] Code formatted (clang-format, rustfmt)
- [ ] Linters pass (clang-tidy, clippy)
- [ ] Commit messages are clear and descriptive
- [ ] Branch is up to date with main

### Creating a Pull Request

1. **Push your changes:**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create PR on GitHub** with:
   - Clear title describing the change
   - Detailed description of what and why
   - Reference to related issues (`Fixes #123`)
   - Screenshots/output if applicable

3. **PR Template** will guide you through required information

4. **Address review feedback** promptly

### Pull Request Template

When creating a PR, include:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] All tests passing
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings
```

## Documentation

### Code Documentation

- **C++**: Use Doxygen-style comments
  ```cpp
  /// Calculate the distance between two points
  /// \param p1 First point
  /// \param p2 Second point
  /// \return Distance between points
  double calculate_distance(const Point& p1, const Point& p2);
  ```

- **Rust**: Use rustdoc comments
  ```rust
  /// Calculates the distance between two points.
  ///
  /// # Arguments
  /// * `p1` - First point
  /// * `p2` - Second point
  ///
  /// # Returns
  /// Distance between the points
  pub fn calculate_distance(p1: &Point, p2: &Point) -> f64 {
      // implementation
  }
  ```

### User Documentation

When adding new features:

- Update relevant `README.md` files
- Add examples in documentation
- Update `QUICKSTART.md` if workflow changes
- Add to `AGENTS.md` if agent-related

### Curriculum Updates

If adding educational content:

- Follow existing week structure
- Include both C++ and Rust implementations
- Add comparative notes
- Create exercises if appropriate

## Review Process

1. **Automated checks** run on every PR (CI/CD)
2. **Code review** by maintainers
3. **Changes requested** may require updates
4. **Approval** by at least one maintainer
5. **Merge** after approval and passing checks

## Getting Help

- **Documentation**: Start with [README.md](README.md) and [QUICKSTART.md](QUICKSTART.md)
- **Issues**: Search existing issues before creating new ones
- **Questions**: Open a discussion or issue with the `question` label
- **Chat**: (Add Discord/Slack link if available)

## Recognition

Contributors are recognized in:
- Git commit history
- Release notes
- Contributors file (if maintained)

Thank you for contributing to Ferric Continuum! 🦀⚙️

