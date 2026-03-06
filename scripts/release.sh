#!/usr/bin/env bash
# =============================================================================
# release.sh — 一键发版脚本
#
# 用法：
#   ./scripts/release.sh patch     # 0.1.0 → 0.1.1
#   ./scripts/release.sh minor     # 0.1.0 → 0.2.0
#   ./scripts/release.sh major     # 0.1.0 → 1.0.0
#   ./scripts/release.sh 0.3.1     # 直接指定版本号
#
# 前置条件：
#   - git 工作区干净（无未提交改动）
#   - 当前分支为 main（可通过 RELEASE_BRANCH 覆盖）
#   - uv 已安装
# =============================================================================

set -euo pipefail

# ---------- 配置 ----------
RELEASE_BRANCH="${RELEASE_BRANCH:-main}"
PYPROJECT="pyproject.toml"
VERSION_FILE="agent/__init__.py"
CHANGELOG="CHANGELOG.md"

# ---------- 颜色 ----------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[release]${NC} $*"; }
ok()    { echo -e "${GREEN}[release]${NC} $*"; }
warn()  { echo -e "${YELLOW}[release]${NC} $*"; }
error() { echo -e "${RED}[release]${NC} $*" >&2; exit 1; }

# ---------- 检查参数 ----------
BUMP="${1:-}"
[ -z "$BUMP" ] && error "用法: $0 <patch|minor|major|x.y.z>"

# ---------- 检查工作区 ----------
if ! git diff --quiet || ! git diff --cached --quiet; then
  error "工作区有未提交的改动，请先 commit 或 stash"
fi

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CURRENT_BRANCH" != "$RELEASE_BRANCH" ]; then
  error "当前分支为 '$CURRENT_BRANCH'，发版须在 '$RELEASE_BRANCH' 分支上执行"
fi

# ---------- 读取当前版本 ----------
CURRENT_VERSION=$(grep -E '^version\s*=' "$PYPROJECT" | head -1 | sed 's/.*= *"\(.*\)"/\1/')
info "当前版本: ${YELLOW}$CURRENT_VERSION${NC}"

# 拆分 major.minor.patch
IFS='.' read -r V_MAJOR V_MINOR V_PATCH <<< "$CURRENT_VERSION"

# ---------- 计算新版本 ----------
case "$BUMP" in
  major) NEW_VERSION="$((V_MAJOR + 1)).0.0" ;;
  minor) NEW_VERSION="${V_MAJOR}.$((V_MINOR + 1)).0" ;;
  patch) NEW_VERSION="${V_MAJOR}.${V_MINOR}.$((V_PATCH + 1))" ;;
  [0-9]*.[0-9]*.[0-9]*)
    NEW_VERSION="$BUMP"
    ;;
  *) error "无效的版本参数: '$BUMP'，请使用 patch/minor/major 或 x.y.z" ;;
esac

info "新版本:  ${GREEN}$NEW_VERSION${NC}"
echo ""
read -r -p "$(echo -e "${YELLOW}确认发布 v${NEW_VERSION}？[y/N]${NC} ")" CONFIRM
[[ "$CONFIRM" =~ ^[Yy]$ ]] || error "已取消"

# ---------- 更新版本号 ----------
info "更新 $PYPROJECT ..."
sed -i "s/^version = \"${CURRENT_VERSION}\"/version = \"${NEW_VERSION}\"/" "$PYPROJECT"

info "更新 $VERSION_FILE ..."
sed -i "s/__version__ = \"${CURRENT_VERSION}\"/__version__ = \"${NEW_VERSION}\"/" "$VERSION_FILE"

# 校验替换成功
UPDATED_VERSION=$(grep -E '^version\s*=' "$PYPROJECT" | head -1 | sed 's/.*= *"\(.*\)"/\1/')
[ "$UPDATED_VERSION" != "$NEW_VERSION" ] && error "pyproject.toml 版本替换失败，请手动检查"

# ---------- 更新 CHANGELOG ----------
TODAY=$(date +%Y-%m-%d)
info "在 $CHANGELOG 中插入 v${NEW_VERSION} 占位段 ..."

# 如果 [Unreleased] 下方还没有对应版本段，则插入占位
if ! grep -q "^\#\# \[${NEW_VERSION}\]" "$CHANGELOG"; then
  sed -i "s/^\#\# \[Unreleased\]/## [Unreleased]\n\n## [${NEW_VERSION}] - ${TODAY}/" "$CHANGELOG"
fi

# 更新底部比较链接
REPO_URL=$(git remote get-url origin | sed 's/git@github.com:/https:\/\/github.com\//' | sed 's/\.git$//')
if grep -q "^\[Unreleased\]:" "$CHANGELOG"; then
  sed -i "s|^\[Unreleased\]:.*|\[Unreleased\]: ${REPO_URL}/compare/v${NEW_VERSION}...HEAD|" "$CHANGELOG"
else
  echo "" >> "$CHANGELOG"
  echo "[Unreleased]: ${REPO_URL}/compare/v${NEW_VERSION}...HEAD" >> "$CHANGELOG"
fi
if ! grep -q "^\[${NEW_VERSION}\]:" "$CHANGELOG"; then
  echo "[${NEW_VERSION}]: ${REPO_URL}/compare/v${CURRENT_VERSION}...v${NEW_VERSION}" >> "$CHANGELOG"
fi

# ---------- 快速 build 校验 ----------
info "运行 uv build 验证打包 ..."
uv build --quiet && ok "build 通过"

# ---------- 提交 & 打 tag ----------
info "提交版本变更 ..."
git add "$PYPROJECT" "$VERSION_FILE" "$CHANGELOG"
git commit -m "release: v${NEW_VERSION}"

info "打 tag v${NEW_VERSION} ..."
git tag -a "v${NEW_VERSION}" -m "Release v${NEW_VERSION}"

# ---------- 推送 ----------
info "推送到 origin ${RELEASE_BRANCH} ..."
git push origin "$RELEASE_BRANCH"
git push origin "v${NEW_VERSION}"

echo ""
ok "================================================================="
ok " 🎉  v${NEW_VERSION} 发布流程已触发！"
ok "================================================================="
echo ""
echo -e "  ${CYAN}GitHub Actions:${NC}"
echo -e "    https://github.com/0verL1nk/LLM_App_Final/actions"
echo ""
echo -e "  ${CYAN}GitHub Release（构建完成后可见）:${NC}"
echo -e "    https://github.com/0verL1nk/LLM_App_Final/releases/tag/v${NEW_VERSION}"
echo ""
echo -e "  ${CYAN}PyPI（发布成功后可见）:${NC}"
echo -e "    https://pypi.org/project/paper-sage/${NEW_VERSION}/"
echo ""
