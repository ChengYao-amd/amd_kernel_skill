#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_NAME="amd-kernel"

CORE_DIRS=(SKILL.md skills references scripts templates)

usage() {
    cat <<'EOF'
AMD Kernel Agent Skill Pack — 安装脚本

用法:
  ./install.sh <mode> <target_project_path>

模式:
  cursor          复制到 <project>/.cursor/skills/amd-kernel/
  claude          复制到 <project>/.claude/skills/amd-kernel/
  cursor-link     符号链接到 .cursor/skills/（开发模式，实时同步）
  claude-link     符号链接到 .claude/skills/（开发模式，实时同步）
  custom <dir>    复制到任意指定目录
  uninstall       从 .cursor/skills/ 和 .claude/skills/ 中移除

选项:
  --with-vendor   同时复制 vendor/ submodule（参考代码库，较大）
  --with-pdf      同时复制 rocm-related-pdf/（AMD 官方 PDF）
  --full          等同于 --with-vendor --with-pdf
  --dry-run       只打印将要执行的操作，不实际执行

示例:
  ./install.sh cursor /home/user/my-project
  ./install.sh claude-link /home/user/my-project
  ./install.sh cursor /home/user/my-project --full
  ./install.sh custom /home/user/my-project/.agent/skills/amd --with-pdf
EOF
    exit 1
}

[[ $# -lt 2 ]] && usage

MODE="$1"
TARGET_PROJECT="$2"
shift 2

WITH_VENDOR=false
WITH_PDF=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-vendor) WITH_VENDOR=true ;;
        --with-pdf)    WITH_PDF=true ;;
        --full)        WITH_VENDOR=true; WITH_PDF=true ;;
        --dry-run)     DRY_RUN=true ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
    shift
done

run_cmd() {
    if $DRY_RUN; then
        echo "[dry-run] $*"
    else
        "$@"
    fi
}

resolve_target() {
    case "$MODE" in
        cursor|cursor-link)
            echo "${TARGET_PROJECT}/.cursor/skills/${SKILL_NAME}" ;;
        claude|claude-link)
            echo "${TARGET_PROJECT}/.claude/skills/${SKILL_NAME}" ;;
        custom)
            echo "${TARGET_PROJECT}" ;;
        uninstall)
            echo "" ;;
        *)
            echo "Unknown mode: $MODE"; usage ;;
    esac
}

do_copy() {
    local dest="$1"
    echo "=== 复制安装到 ${dest} ==="
    run_cmd mkdir -p "$dest"

    for item in "${CORE_DIRS[@]}"; do
        if [[ -d "${SCRIPT_DIR}/${item}" ]]; then
            run_cmd cp -r "${SCRIPT_DIR}/${item}" "$dest/"
        else
            run_cmd cp "${SCRIPT_DIR}/${item}" "$dest/"
        fi
    done

    if $WITH_VENDOR; then
        echo "--- 复制 vendor/ (参考代码库) ---"
        run_cmd cp -r "${SCRIPT_DIR}/vendor" "$dest/"
    fi

    if $WITH_PDF; then
        echo "--- 复制 rocm-related-pdf/ ---"
        run_cmd cp -r "${SCRIPT_DIR}/rocm-related-pdf" "$dest/"
    fi

    echo "=== 安装完成: ${dest} ==="
    echo ""
    echo "文件统计:"
    if ! $DRY_RUN; then
        find "$dest" -type f ! -path '*/vendor/*' ! -path '*/.git/*' | wc -l
        echo "个文件 (不含 vendor)"
    fi
}

do_link() {
    local dest="$1"
    echo "=== 符号链接安装到 ${dest} ==="

    local parent
    parent="$(dirname "$dest")"
    run_cmd mkdir -p "$parent"

    if [[ -e "$dest" ]]; then
        echo "目标已存在: ${dest}"
        echo "请先删除或使用 uninstall 模式"
        exit 1
    fi

    run_cmd ln -s "${SCRIPT_DIR}" "$dest"
    echo "=== 链接完成: ${dest} → ${SCRIPT_DIR} ==="
}

do_uninstall() {
    echo "=== 卸载 ==="
    local cursor_path="${TARGET_PROJECT}/.cursor/skills/${SKILL_NAME}"
    local claude_path="${TARGET_PROJECT}/.claude/skills/${SKILL_NAME}"

    for p in "$cursor_path" "$claude_path"; do
        if [[ -e "$p" ]] || [[ -L "$p" ]]; then
            echo "移除: $p"
            run_cmd rm -rf "$p"
        else
            echo "不存在: $p (跳过)"
        fi
    done
    echo "=== 卸载完成 ==="
}

DEST="$(resolve_target)"

case "$MODE" in
    cursor|claude|custom)
        do_copy "$DEST" ;;
    cursor-link|claude-link)
        do_link "$DEST" ;;
    uninstall)
        do_uninstall ;;
esac
