#!/usr/bin/env bash
# Ferric Continuum - Code Linting Script
# Runs static analysis on C++ and Rust code with optional auto-fix

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FIX_MODE=false
CPP_ONLY=false
RUST_ONLY=false

# Tracking
CPP_ISSUES=0
RUST_ISSUES=0

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --fix)
      FIX_MODE=true
      shift
      ;;
    --cpp-only)
      CPP_ONLY=true
      shift
      ;;
    --rust-only)
      RUST_ONLY=true
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --fix         Automatically fix issues where possible"
      echo "  --cpp-only    Lint only C++ files"
      echo "  --rust-only   Lint only Rust files"
      echo "  -h, --help    Show this help message"
      exit 0
      ;;
    *)
      echo -e "${RED}Unknown option: $1${NC}"
      exit 1
      ;;
  esac
done

# Change to project root
cd "${PROJECT_ROOT}"

# Lint C++ files with clang-tidy
lint_cpp() {
  echo -e "${YELLOW}Linting C++ files with clang-tidy...${NC}"

  # Check if clang-tidy is available
  if ! command -v clang-tidy &> /dev/null; then
    echo -e "${RED}Error: clang-tidy not found. Please install clang-tidy.${NC}"
    return 1
  fi

  # Find all C++ source files
  CPP_FILES=$(find . -type f \
    \( -name "*.cpp" -o -name "*.cc" -o -name "*.cxx" \) \
    ! -path "*/.*" \
    ! -path "*/bazel-*" \
    ! -path "*/build/*" \
    ! -path "*/third_party/*" \
    2>/dev/null || true)

  if [[ -z "${CPP_FILES}" ]]; then
    echo -e "${YELLOW}No C++ source files found.${NC}"
    return 0
  fi

  # Check for compile_commands.json
  if [[ ! -f "compile_commands.json" ]]; then
    echo -e "${YELLOW}Warning: compile_commands.json not found.${NC}"
    echo -e "${YELLOW}Generate it with: bazel run @hedron_compile_commands//:refresh_all${NC}"
    echo -e "${YELLOW}Attempting to lint without compilation database...${NC}"
    COMPILE_DB_FLAG=""
  else
    COMPILE_DB_FLAG="-p ."
  fi

  # Lint each file
  local file_count=0
  local issues_found=0

  while IFS= read -r file; do
    ((file_count++))
    echo -e "${BLUE}Checking: ${file}${NC}"

    if [[ "${FIX_MODE}" == "true" ]]; then
      # Run with auto-fix
      if ! clang-tidy --fix ${COMPILE_DB_FLAG} "${file}" 2>&1 | grep -v "warnings generated" | grep -v "^$"; then
        ((issues_found++))
      fi
    else
      # Run in check mode
      if ! clang-tidy ${COMPILE_DB_FLAG} "${file}" 2>&1 | tee /dev/tty | grep -q "warnings generated"; then
        :
      else
        ((issues_found++))
      fi
    fi
  done <<< "${CPP_FILES}"

  if [[ ${issues_found} -gt 0 ]]; then
    echo -e "${RED}✗ Found issues in ${issues_found} C++ file(s)${NC}"
    CPP_ISSUES=${issues_found}
  else
    echo -e "${GREEN}✓ No issues found in ${file_count} C++ file(s)${NC}"
  fi
}

# Lint Rust files with clippy
lint_rust() {
  echo -e "${YELLOW}Linting Rust files with clippy...${NC}"

  # Check if cargo is available
  if ! command -v cargo &> /dev/null; then
    echo -e "${YELLOW}Cargo not found. Skipping Rust linting.${NC}"
    return 0
  fi

  # Check if clippy is available
  if ! cargo clippy --version &> /dev/null; then
    echo -e "${RED}Error: clippy not found. Install with: rustup component add clippy${NC}"
    return 1
  fi

  # Check if there are any Rust files
  if ! find . -name "Cargo.toml" ! -path "*/.*" ! -path "*/bazel-*" -print -quit | grep -q .; then
    echo -e "${YELLOW}No Rust projects found.${NC}"
    return 0
  fi

  # Run clippy
  local clippy_result=0

  if [[ "${FIX_MODE}" == "true" ]]; then
    # Run with auto-fix
    echo -e "${BLUE}Running clippy with auto-fix...${NC}"
    if cargo clippy --fix --all-targets --all-features --allow-dirty --allow-staged; then
      echo -e "${GREEN}✓ Clippy auto-fix completed${NC}"
    else
      clippy_result=$?
      echo -e "${RED}✗ Clippy found issues that couldn't be auto-fixed${NC}"
      RUST_ISSUES=1
    fi
  else
    # Run in check mode
    echo -e "${BLUE}Running clippy...${NC}"
    if cargo clippy --all-targets --all-features -- -D warnings; then
      echo -e "${GREEN}✓ No clippy issues found${NC}"
    else
      clippy_result=$?
      echo -e "${RED}✗ Clippy found issues${NC}"
      RUST_ISSUES=1
    fi
  fi

  return ${clippy_result}
}

# Main execution
echo -e "${GREEN}=== Ferric Continuum Code Linter ===${NC}"
echo -e "Project root: ${PROJECT_ROOT}"
echo -e "Mode: $([ "${FIX_MODE}" == "true" ] && echo "Fix" || echo "Check")"
echo ""

# Run linters based on flags
OVERALL_RESULT=0

if [[ "${RUST_ONLY}" == "true" ]]; then
  lint_rust || OVERALL_RESULT=$?
elif [[ "${CPP_ONLY}" == "true" ]]; then
  lint_cpp || OVERALL_RESULT=$?
else
  lint_cpp || OVERALL_RESULT=$?
  echo ""
  lint_rust || OVERALL_RESULT=$?
fi

# Summary
echo ""
echo -e "${GREEN}=== Linting Summary ===${NC}"

if [[ $((CPP_ISSUES + RUST_ISSUES)) -eq 0 ]]; then
  echo -e "${GREEN}✓ All checks passed!${NC}"
  exit 0
else
  if [[ "${FIX_MODE}" == "true" ]]; then
    echo -e "${YELLOW}Some issues were fixed, but others require manual attention.${NC}"
    echo -e "${YELLOW}Review the changes and address remaining issues.${NC}"
  else
    echo -e "${RED}Issues found. Run with --fix to automatically fix some issues.${NC}"
  fi
  exit 1
fi
