# Support

Thank you for using Ferric Continuum! This document provides information on how to get help.

## 📚 Documentation

Before asking for help, please check our documentation:

- **[README.md](../README.md)** - Project overview, quick start, and examples
- **[AGENTS.md](../AGENTS.md)** - Agent architecture and 10-week curriculum
- **[ENGINEERING.md](../ENGINEERING.md)** - Coding standards and best practices
- **[CONTRIBUTING.md](../CONTRIBUTING.md)** - How to contribute

## 🐛 Found a Bug?

If you've found a bug, please:

1. **Search existing issues** to see if it's already reported
2. **Check if it's fixed** in the latest version
3. **Create a bug report** using our [bug report template](https://github.com/USERNAME/ferric_continuum/issues/new?template=bug_report.md)

Include:
- Steps to reproduce
- Expected vs. actual behavior
- Your environment (OS, Bazel version, etc.)
- Error messages and logs

## 💡 Have a Feature Request?

We welcome feature requests! Please:

1. **Search existing issues** to avoid duplicates
2. **Create a feature request** using our [feature request template](https://github.com/USERNAME/ferric_continuum/issues/new?template=feature_request.md)

Include:
- Clear description of the feature
- Use cases and benefits
- Proposed implementation (optional)

## ❓ Have a Question?

For questions about using Ferric Continuum:

### 1. Check the Documentation First

Most questions are answered in our docs:
- Installation & Quick Start → [README.md](../README.md)
- Architecture questions → [AGENTS.md](../AGENTS.md)
- Coding standards → [ENGINEERING.md](../ENGINEERING.md)
- Contributing process → [CONTRIBUTING.md](../CONTRIBUTING.md)

### 2. Search Existing Issues and Discussions

Someone may have already asked your question:
- [GitHub Issues](https://github.com/USERNAME/ferric_continuum/issues)
- [GitHub Discussions](https://github.com/USERNAME/ferric_continuum/discussions)

### 3. Ask in GitHub Discussions

For questions, ideas, and general discussion:
- [GitHub Discussions](https://github.com/USERNAME/ferric_continuum/discussions)

Use the appropriate category:
- **Q&A** - General questions
- **Ideas** - Feature ideas and brainstorming
- **Show and Tell** - Share your projects using Ferric Continuum
- **General** - Everything else

### 4. Create an Issue

If your question reveals a documentation gap or bug, please create an issue.

## 🔒 Security Issues

**DO NOT** create public issues for security vulnerabilities.

Please report security issues privately by emailing:
- **[INSERT SECURITY EMAIL ADDRESS]**

See [SECURITY.md](../SECURITY.md) for details.

## 💬 Community

Join our community:

- **GitHub Discussions**: [Discussions](https://github.com/USERNAME/ferric_continuum/discussions)
- **Issues**: [Issues](https://github.com/USERNAME/ferric_continuum/issues)

(Add links to Discord, Slack, or other community platforms if available)

## ⚡ Quick Help for Common Issues

### Build Failures

```bash
# Clean build artifacts
bazel clean --expunge

# Rebuild everything
bazel build //...

# Check Bazel version
bazel --version  # Should be 8.4.2+
```

### Test Failures

```bash
# Run tests with verbose output
bazel test //... --test_output=all

# Run specific test
bazel test //ferric_continuum/foundation:value_semantics_test_cc --test_output=all
```

### Dependency Issues

```bash
# Sync dependencies
bazel sync

# Update lock file
bazel mod tidy
```

### Format/Lint Issues

```bash
# Format all code
./scripts/format.sh

# Run linters
./scripts/lint.sh
```

## 📖 Additional Resources

- **Bazel Documentation**: https://bazel.build/docs
- **Abseil Documentation**: https://abseil.io/docs/
- **Rust Documentation**: https://doc.rust-lang.org/
- **Google C++ Style Guide**: https://google.github.io/styleguide/cppguide.html
- **Rust API Guidelines**: https://rust-lang.github.io/api-guidelines/

## 🤝 Contributing

Want to help improve Ferric Continuum? See [CONTRIBUTING.md](../CONTRIBUTING.md)!

## 📧 Contact

- **General inquiries**: [INSERT CONTACT EMAIL]
- **Security issues**: [INSERT SECURITY EMAIL]

---

Thank you for being part of the Ferric Continuum community! 🦀⚙️

