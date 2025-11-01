#!/bin/bash
# Verification script for GitHub hosting setup

set -e

echo "=================================================="
echo "Ferric Continuum - GitHub Setup Verification"
echo "=================================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
PASSED=0
FAILED=0
WARNINGS=0

check_file() {
    local file=$1
    local description=$2
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $description"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}✗${NC} $description (missing: $file)"
        ((FAILED++))
        return 1
    fi
}

check_placeholder() {
    local file=$1
    local placeholder=$2
    if grep -q "$placeholder" "$file" 2>/dev/null; then
        echo -e "${YELLOW}⚠${NC} Placeholder '$placeholder' found in $file"
        ((WARNINGS++))
        return 1
    fi
    return 0
}

echo "Checking core documentation files..."
echo "-----------------------------------"
check_file "LICENSE" "LICENSE file"
check_file "README.md" "README.md"
check_file "CONTRIBUTING.md" "Contributing guidelines"
check_file "CODE_OF_CONDUCT.md" "Code of Conduct"
check_file "SECURITY.md" "Security policy"
check_file "CHANGELOG.md" "Changelog"
check_file ".gitignore" ".gitignore"
check_file ".gitattributes" ".gitattributes"
echo ""

echo "Checking GitHub configuration files..."
echo "--------------------------------------"
check_file ".github/CODEOWNERS" "CODEOWNERS"
check_file ".github/SUPPORT.md" "Support documentation"
check_file ".github/FUNDING.yml" "Funding configuration"
check_file ".github/dependabot.yml" "Dependabot configuration"
check_file ".github/labeler.yml" "Labeler configuration"
check_file ".github/release-drafter.yml" "Release drafter config"
echo ""

echo "Checking issue templates..."
echo "---------------------------"
check_file ".github/ISSUE_TEMPLATE/bug_report.md" "Bug report template"
check_file ".github/ISSUE_TEMPLATE/feature_request.md" "Feature request template"
check_file ".github/ISSUE_TEMPLATE/config.yml" "Issue template config"
echo ""

echo "Checking PR templates..."
echo "------------------------"
check_file ".github/PULL_REQUEST_TEMPLATE/pull_request_template.md" "Pull request template"
echo ""

echo "Checking GitHub Actions workflows..."
echo "------------------------------------"
check_file ".github/workflows/ci.yml" "CI workflow"
check_file ".github/workflows/release.yml" "Release workflow"
check_file ".github/workflows/release-drafter.yml" "Release drafter workflow"
check_file ".github/workflows/labeler.yml" "Labeler workflow"
echo ""

echo "Checking for placeholder text..."
echo "--------------------------------"
PLACEHOLDER_FOUND=false

# Check for USERNAME placeholder
for file in README.md CONTRIBUTING.md .github/SUPPORT.md .github/CODEOWNERS .github/ISSUE_TEMPLATE/config.yml; do
    if [ -f "$file" ]; then
        check_placeholder "$file" "USERNAME" || PLACEHOLDER_FOUND=true
    fi
done

# Check for email placeholders
check_placeholder "CODE_OF_CONDUCT.md" "INSERT CONTACT EMAIL" || PLACEHOLDER_FOUND=true
check_placeholder "SECURITY.md" "INSERT SECURITY EMAIL" || PLACEHOLDER_FOUND=true
check_placeholder ".github/SUPPORT.md" "INSERT CONTACT EMAIL" || PLACEHOLDER_FOUND=true
echo ""

echo "Checking build system..."
echo "------------------------"
if command -v bazel &> /dev/null; then
    BAZEL_VERSION=$(bazel --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
    echo -e "${GREEN}✓${NC} Bazel installed (version $BAZEL_VERSION)"
    ((PASSED++))
else
    echo -e "${RED}✗${NC} Bazel not found"
    ((FAILED++))
fi

if [ -f "MODULE.bazel" ]; then
    echo -e "${GREEN}✓${NC} MODULE.bazel exists"
    ((PASSED++))
else
    echo -e "${RED}✗${NC} MODULE.bazel missing"
    ((FAILED++))
fi
echo ""

echo "Checking code quality tools..."
echo "------------------------------"
if command -v clang-format &> /dev/null; then
    echo -e "${GREEN}✓${NC} clang-format installed"
    ((PASSED++))
else
    echo -e "${YELLOW}⚠${NC} clang-format not found"
    ((WARNINGS++))
fi

if command -v rustfmt &> /dev/null; then
    echo -e "${GREEN}✓${NC} rustfmt installed"
    ((PASSED++))
else
    echo -e "${YELLOW}⚠${NC} rustfmt not found"
    ((WARNINGS++))
fi

if [ -f ".clang-format" ]; then
    echo -e "${GREEN}✓${NC} .clang-format exists"
    ((PASSED++))
fi

if [ -f "rustfmt.toml" ]; then
    echo -e "${GREEN}✓${NC} rustfmt.toml exists"
    ((PASSED++))
fi
echo ""

echo "Verifying .gitignore..."
echo "-----------------------"
if grep -q "bazel-" .gitignore 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Bazel artifacts ignored"
    ((PASSED++))
else
    echo -e "${RED}✗${NC} Bazel artifacts not in .gitignore"
    ((FAILED++))
fi

if grep -q "__pycache__" .gitignore 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Python cache ignored"
    ((PASSED++))
else
    echo -e "${YELLOW}⚠${NC} Python cache not in .gitignore"
    ((WARNINGS++))
fi
echo ""

echo "Checking test execution..."
echo "--------------------------"
if bazel test //... --test_summary=short 2>&1 | grep -q "//"; then
    echo -e "${GREEN}✓${NC} Tests can be run with Bazel"
    ((PASSED++))
else
    echo -e "${YELLOW}⚠${NC} Test execution could not be verified"
    ((WARNINGS++))
fi
echo ""

echo "=================================================="
echo "Verification Summary"
echo "=================================================="
echo -e "${GREEN}Passed:${NC}   $PASSED"
echo -e "${RED}Failed:${NC}   $FAILED"
echo -e "${YELLOW}Warnings:${NC} $WARNINGS"
echo ""

if [ $FAILED -eq 0 ]; then
    if [ $WARNINGS -eq 0 ]; then
        echo -e "${GREEN}✓ All checks passed! Ready for GitHub.${NC}"
        echo ""
        echo "Next steps:"
        echo "1. Review GITHUB_HOSTING_SETUP.md for final configuration"
        echo "2. Replace USERNAME placeholders with your GitHub username"
        echo "3. Add contact emails to CODE_OF_CONDUCT.md and SECURITY.md"
        echo "4. Push to GitHub: git push origin main"
        exit 0
    else
        echo -e "${YELLOW}⚠ Setup complete with warnings. Please review warnings above.${NC}"
        echo ""
        echo "Action items:"
        echo "1. Replace USERNAME placeholders if present"
        echo "2. Add contact emails if placeholders found"
        echo "3. Install missing tools (clang-format, rustfmt) if needed"
        exit 0
    fi
else
    echo -e "${RED}✗ Setup incomplete. Please fix the issues above.${NC}"
    exit 1
fi

