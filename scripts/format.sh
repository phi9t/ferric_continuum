#!/bin/bash
# Ferric Continuum - Code Formatting Script
# Formats all C++ and Rust code in the repository

set -eu -o pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CHECK_ONLY=false
CPP_ONLY=false
RUST_ONLY=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --check)
      CHECK_ONLY=true
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
      echo "  --check       Check formatting without modifying files (CI mode)"
      echo "  --cpp-only    Format only C++ files"
      echo "  --rust-only   Format only Rust files"
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

# Track formatting status
CPP_FORMAT_NEEDED=0
RUST_FORMAT_NEEDED=0

# Format C++ files
format_cpp() {
  echo -e "${YELLOW}Formatting C++ files...${NC}"

  # Check if clang-format is available
  if ! command -v clang-format &> /dev/null; then
    echo -e "${RED}Error: clang-format not found. Please install clang-format.${NC}"
    return 1
  fi

  # Find all C++ files
  CPP_FILES=$(find . -type f \
    \( -name "*.cpp" -o -name "*.cc" -o -name "*.cxx" \
    -o -name "*.h" -o -name "*.hpp" -o -name "*.hxx" \) \
    ! -path "*/.*" \
    ! -path "*/bazel-*" \
    ! -path "*/build/*" \
    ! -path "*/third_party/*" \
    2>/dev/null || true)

  if [[ -z "${CPP_FILES}" ]]; then
    echo -e "${YELLOW}No C++ files found.${NC}"
    return 0
  fi

  if [[ "${CHECK_ONLY}" == "true" ]]; then
    # Check mode - don't modify files
    if echo "${CPP_FILES}" | xargs clang-format --dry-run --Werror 2>&1; then
      echo -e "${GREEN}✓ All C++ files are properly formatted${NC}"
    else
      echo -e "${RED}✗ Some C++ files need formatting${NC}"
      CPP_FORMAT_NEEDED=1
    fi
  else
    # Format mode - modify files
    echo "${CPP_FILES}" | xargs clang-format -i
    echo -e "${GREEN}✓ Formatted $(echo "${CPP_FILES}" | wc -l) C++ files${NC}"
  fi
}

# Format Rust files
format_rust() {
  echo -e "${YELLOW}Formatting Rust files...${NC}"

  # Check if cargo is available
  if ! command -v cargo &> /dev/null; then
    echo -e "${YELLOW}Cargo not found. Skipping Rust formatting.${NC}"
    return 0
  fi

  # Check if rustfmt is available
  if ! cargo fmt --version &> /dev/null; then
    echo -e "${RED}Error: rustfmt not found. Install with: rustup component add rustfmt${NC}"
    return 1
  fi

  # Check if there are any Rust files
  if ! find . -name "Cargo.toml" ! -path "*/.*" ! -path "*/bazel-*" -print -quit | grep -q .; then
    echo -e "${YELLOW}No Rust projects found.${NC}"
    return 0
  fi

  if [[ "${CHECK_ONLY}" == "true" ]]; then
    # Check mode
    if cargo fmt --all -- --check 2>&1; then
      echo -e "${GREEN}✓ All Rust files are properly formatted${NC}"
    else
      echo -e "${RED}✗ Some Rust files need formatting${NC}"
      RUST_FORMAT_NEEDED=1
    fi
  else
    # Format mode
    cargo fmt --all
    echo -e "${GREEN}✓ Formatted all Rust files${NC}"
  fi
}

# Main execution
echo -e "${GREEN}=== Ferric Continuum Code Formatter ===${NC}"
echo -e "Project root: ${PROJECT_ROOT}"
echo -e "Mode: $([ "${CHECK_ONLY}" == "true" ] && echo "Check" || echo "Format")"
echo ""

# Run formatters based on flags
if [[ "${RUST_ONLY}" == "true" ]]; then
  format_rust
elif [[ "${CPP_ONLY}" == "true" ]]; then
  format_cpp
else
  format_cpp
  format_rust
fi

# Exit with appropriate code
if [[ "${CHECK_ONLY}" == "true" ]]; then
  if [[ $((CPP_FORMAT_NEEDED + RUST_FORMAT_NEEDED)) -gt 0 ]]; then
    echo ""
    echo -e "${RED}Formatting check failed. Run './scripts/format.sh' to fix.${NC}"
    exit 1
  else
    echo ""
    echo -e "${GREEN}All files are properly formatted!${NC}"
    exit 0
  fi
else
  echo ""
  echo -e "${GREEN}Formatting complete!${NC}"
  exit 0
fi
