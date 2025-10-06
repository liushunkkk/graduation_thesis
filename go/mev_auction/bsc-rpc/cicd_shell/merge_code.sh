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

print_green "----> 合并分支阶段"

print_yellow "当前目录：$(pwd)" # builds/shengweifeng/bsc-rpc
print_yellow "在上级目录创建clone_project目录，clone project"
cd ../ && pwd && rm -rf clone_project && mkdir clone_project && cd clone_project && pwd  || exit 1
git clone https://gitlab.com/shengweifeng/bsc-rpc.git || exit 1
cd bsc-rpc && pwd || exit 1
git fetch origin || exit 1

print_yellow "切换到源分支 $CI_MERGE_REQUEST_SOURCE_BRANCH_NAME"
git checkout -b $CI_MERGE_REQUEST_SOURCE_BRANCH_NAME origin/$CI_MERGE_REQUEST_SOURCE_BRANCH_NAME || exit 1
print_yellow "$(git log -n 10 --pretty=format:"%H")" || exit 1

print_yellow "切换到目标分支 $CI_MERGE_REQUEST_TARGET_BRANCH_NAME"
git checkout -b $CI_MERGE_REQUEST_TARGET_BRANCH_NAME origin/$CI_MERGE_REQUEST_TARGET_BRANCH_NAME || exit 1
print_yellow "$(git log -n 10 --pretty=format:"%H")" || exit 1

print_yellow "合并代码 源->目标"
git pull origin $CI_MERGE_REQUEST_SOURCE_BRANCH_NAME --no-rebase --no-edit --commit || exit 1
print_yellow "$(git log -n 10 --pretty=format:"%H")" || exit 1

print_green "<---- 合并分支阶段结束"
