# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure with C++ and Rust foundation libraries
- Comprehensive foundation examples (value semantics, move semantics, parameter passing, smart pointers, constructor rules)
- Bazel 8.4.2 build system with Bzlmod
- Abseil C++ library integration
- Rust tracing and logging integration
- 74 unit tests with 100% pass rate
- Structured logging for all code
- Code formatting tools (clang-format, rustfmt)
- Code linting tools (clang-tidy, clippy)
- Documentation: README, QUICKSTART, AGENTS, ENGINEERING
- GitHub CI/CD workflows
- Contributing guidelines and Code of Conduct
- MIT License

### Foundation Libraries (C++ and Rust)

#### Value Semantics
- Copy constructors and assignment operators
- Deep copy vs. shallow copy demonstrations
- Value vs. reference semantics comparison

#### Move Semantics
- Move constructors and assignment operators
- Efficient resource transfer patterns
- std::move and performance optimization

#### Parameter Passing
- Pass by value, reference, const reference
- Move parameter optimization
- Best practices for function parameters

#### Smart Pointers & RAII
- std::unique_ptr, std::shared_ptr, std::weak_ptr (C++)
- Box, Rc, Arc, Weak (Rust)
- Automatic resource management
- Memory leak prevention

#### Constructor Rules
- Rule of Zero
- Rule of Five (C++)
- Default, copy, move constructors and destructors
- Copy and move assignment operators
- Move-only types

### Development Tools
- Format scripts (`scripts/format.sh`)
- Lint scripts (`scripts/lint.sh`)
- Test coverage reporting

### Documentation
- Comprehensive README with badges
- QUICKSTART guide
- AGENTS architecture and 10-week curriculum
- ENGINEERING standards and best practices
- CONTRIBUTING guide
- CODE_OF_CONDUCT
- SECURITY policy
- TEST_COVERAGE_SUMMARY

### CI/CD
- GitHub Actions CI workflow
- GitHub Actions release workflow
- Dependabot configuration
- Code owners
- Issue templates (bug report, feature request)
- Pull request template

## [0.1.0] - YYYY-MM-DD (Future)

### Planned
- Agent implementations (BuildMind, PerfSmith, AsyncHermes, SpackSentinel, ContinuumSupervisor)
- Matrix multiplication examples
- Parallel algorithms
- Distributed computing primitives
- gRPC integration
- Benchmarking with Google Benchmark and Criterion
- Performance profiling tools
- Observability dashboard
- Spack integration for HPC libraries

---

## Version History

The version history will be maintained as releases are created.

### Version Format

- **Major (X.0.0)**: Breaking changes
- **Minor (0.X.0)**: New features, backward compatible
- **Patch (0.0.X)**: Bug fixes, backward compatible

### Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting)
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `test:` Test additions or changes
- `build:` Build system changes
- `ci:` CI configuration changes
- `chore:` Other changes (e.g., dependency updates)

---

[Unreleased]: https://github.com/USERNAME/ferric_continuum/compare/v0.1.0...HEAD

