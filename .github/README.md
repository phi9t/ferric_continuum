# .github Directory Structure

This directory contains all GitHub-specific configuration files for the Ferric Continuum repository.

## 📁 Directory Structure

```
.github/
├── CODEOWNERS              # Code ownership configuration
├── FUNDING.yml             # Sponsorship/funding configuration
├── README.md               # This file
├── SUPPORT.md              # Support and help resources
├── dependabot.yml          # Automated dependency updates
├── labeler.yml             # Automatic PR labeling rules
├── release-drafter.yml     # Release notes configuration
│
├── ISSUE_TEMPLATE/         # Issue templates
│   ├── bug_report.md       # Bug report template
│   ├── config.yml          # Issue template configuration
│   └── feature_request.md  # Feature request template
│
├── PULL_REQUEST_TEMPLATE/  # PR templates
│   └── pull_request_template.md
│
└── workflows/              # GitHub Actions workflows
    ├── ci.yml              # Continuous Integration
    ├── labeler.yml         # PR auto-labeling
    ├── release-drafter.yml # Release notes drafting
    └── release.yml         # Release automation
```

## 🔧 Configuration Files

### CODEOWNERS
Defines who owns which parts of the codebase and who should review PRs.

**Usage:** Automatically requests reviews from code owners when PRs modify their code.

**Configuration:**
```
# C++ code
*.cc @USERNAME
*.hh @USERNAME

# Rust code
*.rs @USERNAME
```

### FUNDING.yml
Configures sponsorship and funding options displayed on the repository.

**Usage:** Enables GitHub Sponsors button and links to funding platforms.

**Supported platforms:** GitHub Sponsors, Patreon, Open Collective, Ko-fi, etc.

### SUPPORT.md
Provides support and help information for users.

**Usage:** Linked from issues page, helps users find help before creating issues.

**Includes:**
- Documentation links
- How to report bugs
- How to ask questions
- Community resources

### dependabot.yml
Configures Dependabot for automated dependency updates.

**Usage:** Creates PRs to update dependencies weekly.

**Monitors:**
- GitHub Actions (weekly)
- Python packages (weekly)
- Rust crates (weekly)

### labeler.yml
Defines rules for automatically labeling PRs based on changed files.

**Usage:** Automatically adds labels like "C++", "Rust", "Documentation" to PRs.

**Example rules:**
```yaml
'C++':
  - '**/*.cc'
  - '**/*.hh'

'Rust':
  - '**/*.rs'
```

### release-drafter.yml
Configures automatic release notes generation.

**Usage:** Drafts release notes based on merged PRs and their labels.

**Categories:**
- 🚀 Features
- 🐛 Bug Fixes
- 📚 Documentation
- 🧰 Maintenance
- etc.

## 📋 Issue Templates

### bug_report.md
Template for reporting bugs with structured sections.

**Sections:**
- Description
- Steps to reproduce
- Expected vs actual behavior
- Environment details
- Code samples

### feature_request.md
Template for requesting new features.

**Sections:**
- Problem description
- Proposed solution
- Alternatives considered
- Implementation details (C++ and Rust)
- Benefits and drawbacks

### config.yml
Configures issue template behavior and links.

**Features:**
- Disables blank issues
- Links to Discussions for questions
- Links to Documentation
- Links to Security policy

## 📝 PR Template

### pull_request_template.md
Template for pull requests with comprehensive checklist.

**Sections:**
- Description and type of change
- Related issues
- Changes made
- Language(s) affected
- Testing performed
- Code quality checklist
- Breaking changes
- Performance impact

## 🤖 GitHub Actions Workflows

### ci.yml (Main CI Workflow)
Runs on every push and PR to main/develop branches.

**Jobs:**
1. **build-and-test** (Ubuntu + macOS)
   - Builds all C++, Rust, Python targets
   - Runs all 74 tests
   - Caches Bazel artifacts
   - Generates test summary

2. **format-check**
   - Checks C++ formatting (clang-format)
   - Checks Rust formatting (rustfmt)
   - Fails if code is not formatted

3. **lint**
   - Runs Rust clippy
   - Reports linting issues

4. **coverage**
   - Generates test coverage
   - Reports coverage metrics

**Triggers:** Push to main/develop, PRs

**Duration:** ~10-15 minutes

### release.yml (Release Workflow)
Runs when a version tag is pushed (e.g., v1.0.0).

**Steps:**
1. Checkout code
2. Build all targets
3. Run all tests
4. Generate changelog
5. Create GitHub release

**Triggers:** Push tags matching `v*`

**Output:** GitHub release with notes

### release-drafter.yml (Release Drafter)
Maintains a draft release with automatic notes.

**Function:**
- Updates draft release on every PR merge
- Categorizes changes by label
- Suggests version bump (major/minor/patch)
- Lists contributors

**Triggers:** Push to main, PR opened/synced

### labeler.yml (PR Labeler)
Automatically labels PRs based on changed files.

**Function:**
- Adds language labels (C++, Rust, Python)
- Adds component labels (Foundation, Agents)
- Adds type labels (Documentation, Tests, CI/CD)

**Triggers:** PR opened/reopened/synchronized

## 🔒 Security

### Security Scanning
- **Dependabot alerts:** Enabled for dependency vulnerabilities
- **Secret scanning:** Detects accidentally committed secrets
- **Code scanning:** Can be enabled for additional security checks

### Branch Protection
Recommended settings for `main` branch:
- Require PR reviews
- Require status checks (build-and-test, format-check, lint)
- Require branches to be up to date
- No force pushes

## 📊 Monitoring

### Actions Tab
View workflow runs: `https://github.com/USERNAME/ferric_continuum/actions`

**Metrics:**
- Success rate
- Duration
- Artifact size
- Cache hit rate

### Insights Tab
View repository insights: `https://github.com/USERNAME/ferric_continuum/pulse`

**Metrics:**
- Contributors
- Commits
- PRs
- Issues
- Traffic

## 🎯 Best Practices

### Workflow Optimization
1. **Cache Bazel artifacts** - Speeds up builds
2. **Run tests in parallel** - Reduces CI time
3. **Fail fast** - Stop on first failure
4. **Matrix builds** - Test on multiple OS/versions

### Issue Management
1. **Use templates** - Ensures complete information
2. **Label appropriately** - Helps with triage
3. **Link to PRs** - Tracks work
4. **Close stale issues** - Keeps repository clean

### PR Reviews
1. **Require reviews** - Maintain code quality
2. **Use code owners** - Right people review
3. **Check CI status** - All green before merge
4. **Squash merge** - Clean history

## 🔄 Update Process

### When to Update

**Workflows:** When CI requirements change
**Templates:** When contribution process changes
**Dependabot:** When adding new dependency sources
**Labeler:** When adding new components/languages

### How to Update

1. Edit files in `.github/`
2. Test changes on a branch
3. Create PR with changes
4. Review and merge

## 📚 Resources

- **GitHub Actions docs:** https://docs.github.com/en/actions
- **Dependabot docs:** https://docs.github.com/en/code-security/dependabot
- **Issue templates:** https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests
- **CODEOWNERS:** https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners

## 🆘 Troubleshooting

### Workflow Failures
```bash
# Check workflow logs in GitHub Actions tab
# Common issues:
- Cache corruption → Re-run workflow
- Dependency issues → Update MODULE.bazel
- Flaky tests → Fix or mark as known flaky
```

### Dependabot Issues
```bash
# If Dependabot PRs fail:
1. Check CI logs
2. Update dependencies manually if needed
3. Merge working updates first
```

### Label Issues
```bash
# If auto-labeling not working:
1. Check labeler.yml syntax
2. Verify workflow is enabled
3. Check PR file patterns match rules
```

---

**Maintained by:** Repository Maintainers
**Last Updated:** 2025-11-01
**Questions?** See [SUPPORT.md](SUPPORT.md)

