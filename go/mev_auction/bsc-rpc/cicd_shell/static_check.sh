#!/usr/bin/env bash

print_green() {
  printf "\033[0;32m%s\033[0m\n" "$1"
}
print_red() {
  printf "\033[0;31m%s\033[0m\n" "$1"
}
print_yellow() {
  printf "\033[0;33m%s\033[0m\n" "$1"
}

print_green "----> 代码静态检查阶段"
print_yellow "当前目录：$(pwd)"

print_yellow "进入项目目录"
cd ../ && pwd && cd clone_project/bsc-rpc  && pwd || exit 1

print_yellow "当前分支: $(git rev-parse --abbrev-ref HEAD)"
print_yellow "当前提交ID：$(git log -n 10 --pretty=format:"%H")"

print_yellow "检查 golangci-lint..."
golangci-lint --version
print_yellow "golangci-lint 检查完成。"
print_yellow "开始进行代码静态检查..."

golangci-lint run -v --timeout=10m --color always --out-format junit-xml:./code-lint-report.xml  core/txpool/bundlepool/... eth/scutum/... invalid_tx/... portal/... push/... relay/... validator/...

print_green "<---- 代码静态检查完成"