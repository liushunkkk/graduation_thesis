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

#print_green "----> 合并分支阶段"
#
#print_yellow "当前目录：$(pwd)"
#
#print_yellow "进入项目目录"
#cd ../ && pwd && cd clone_project/bsc-rpc  && pwd || exit 1
#
#print_yellow "当前分支: $(git rev-parse --abbrev-ref HEAD)"
#print_yellow "当前提交ID：$(git log -n 10 --pretty=format:"%H")"
#
##print_yellow "合并代码 源->目标"
##git pull origin $CI_MERGE_REQUEST_SOURCE_BRANCH_NAME --no-rebase --no-edit --commit || exit 1
##print_yellow "$(git log -n 10 --pretty=format:"%H")" || exit 1
#
#print_green "<---- 合并分支阶段成功"

print_green "----> 单元测试阶段"

print_yellow "当前目录：$(pwd)"
print_yellow "进入项目目录"
cd ../ && pwd && cd clone_project/bsc-rpc  && pwd || exit 1

print_yellow "当前分支: $(git rev-parse --abbrev-ref HEAD)"
print_yellow "当前提交ID：$(git log -n 10 --pretty=format:"%H")"

print_yellow "安装go-junit-report"
go install github.com/jstemmer/go-junit-report@latest || exit 1
print_yellow "开始执行单元测试..."
#go test -v -cover -coverprofile=coverage.out  ./common/ms/... ./core/txpool/bundlepool/bundlepool_test.go ./invalid_tx/... -gcflags "all=-N -l" 2>&1 | go-junit-report > unit-report.xml
go test -v -cover -coverprofile=coverage.out ./invalid_tx/... -gcflags "all=-N -l" 2>&1 | go-junit-report > unit-report.xml
print_green "<---- 单元测试报告生成完成"
