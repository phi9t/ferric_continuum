# Security Policy

## Supported Versions

We currently support the following versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| main    | :white_check_mark: |
| develop | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please report it responsibly:

### How to Report

**Please DO NOT create a public GitHub issue for security vulnerabilities.**

Instead, please report security vulnerabilities by emailing:

📧 **[INSERT SECURITY EMAIL ADDRESS]**

### What to Include

Please include the following information in your report:

1. **Description** - Clear description of the vulnerability
2. **Impact** - Potential impact and attack scenario
3. **Steps to Reproduce** - Detailed steps to reproduce the issue
4. **Affected Versions** - Which versions are affected
5. **Proposed Fix** - (Optional) Suggestions for fixing the issue
6. **Your Contact Info** - How we can reach you for follow-up

### Example Report

```
Subject: [SECURITY] Buffer overflow in matrix multiplication

Description:
A buffer overflow vulnerability exists in the C++ matrix multiplication
code when handling matrices larger than MAX_SIZE.

Impact:
Could lead to arbitrary code execution in applications using this library.

Steps to Reproduce:
1. Create a matrix with dimensions > MAX_SIZE
2. Call matrix_multiply()
3. Observe segmentation fault

Affected Versions:
All versions prior to v1.0.0

Proposed Fix:
Add bounds checking before allocation in matrix_mul.cc:123
```

## Response Timeline

- **Initial Response**: Within 48 hours
- **Confirmation**: Within 1 week
- **Fix Timeline**: Depends on severity
  - Critical: 1-7 days
  - High: 7-14 days
  - Medium: 14-30 days
  - Low: 30-90 days

## Disclosure Policy

We follow **coordinated disclosure**:

1. You report a vulnerability privately
2. We confirm and investigate
3. We develop and test a fix
4. We release a security patch
5. We publicly disclose the issue (with credit to reporter, if desired)

### Public Disclosure Timeline

- **Critical vulnerabilities**: 7-14 days after patch release
- **Non-critical vulnerabilities**: 30 days after patch release

## Security Best Practices

When using Ferric Continuum:

### Build Security

- **Pin Bazel version** - Use `.bazelversion` to ensure reproducible builds
- **Verify dependencies** - Review `MODULE.bazel` and lock files
- **Use hermetic builds** - Bazel's sandboxing provides isolation

### Code Security

- **Memory Safety** - Rust code is memory-safe by default
- **C++ Guidelines** - Follow C++ Core Guidelines for safe C++ code
- **Input Validation** - Always validate external inputs
- **Error Handling** - Use proper error handling (not panics/crashes)

### Dependency Security

```bash
# Check for vulnerable dependencies
bazel query "deps(//...)" 

# Update dependencies regularly
bazel sync --configure
```

### Runtime Security

- **Least Privilege** - Run with minimal required permissions
- **Sandboxing** - Use OS-level sandboxing when available
- **Logging** - Enable structured logging for audit trails
- **Monitoring** - Monitor for suspicious activity

## Known Security Considerations

### C++ Code

- **Raw Pointers** - Minimize use of raw pointers; prefer smart pointers
- **Buffer Management** - Use `std::vector` and `std::string` over raw arrays
- **Integer Overflow** - Check for overflow in arithmetic operations
- **Uninitialized Memory** - Initialize all variables

### Rust Code

- **Unsafe Blocks** - Minimize `unsafe` code; document thoroughly
- **Panic Safety** - Avoid panics in library code
- **FFI Boundaries** - Carefully validate data at C++/Rust boundaries

### Build System

- **External Dependencies** - All external dependencies are fetched via Bazel
- **Reproducible Builds** - Use locked dependency versions
- **Supply Chain** - Verify dependency sources

## Security Updates

Security updates are released as:

- **Patch releases** - For supported versions
- **Security advisories** - Via GitHub Security Advisories
- **Changelog entries** - Documented in CHANGELOG.md

Subscribe to security notifications:
- Watch this repository for security advisories
- Enable GitHub security alerts
- Follow release notes

## Bug Bounty

Currently, we do not have a formal bug bounty program. However, we deeply appreciate security researchers' efforts and will:

- Acknowledge your contribution (unless you prefer to remain anonymous)
- Credit you in release notes
- Provide swag/stickers (when available)

## Security Checklist for Contributors

Before submitting code:

- [ ] No hardcoded credentials or secrets
- [ ] Input validation for external data
- [ ] Proper error handling
- [ ] No use of deprecated/unsafe functions
- [ ] Memory safety verified (C++ code)
- [ ] Minimal use of `unsafe` (Rust code)
- [ ] Security implications documented
- [ ] Tests include edge cases and error conditions

## Contact

For security-related questions (non-vulnerabilities):
- Open a GitHub Discussion with the `security` label
- Email: [INSERT GENERAL CONTACT EMAIL]

For security vulnerabilities:
- Email: [INSERT SECURITY EMAIL ADDRESS]

---

Thank you for helping keep Ferric Continuum secure! 🔒

