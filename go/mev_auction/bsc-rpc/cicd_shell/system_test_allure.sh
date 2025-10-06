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

print_yellow "当前目录：$(pwd)"

print_green "---> 生成测试报告"
cd .. && pwd &&  cd blockrazor_system_test/scutum_test || exit 1
pwd
print_yellow " allure"
allure --version || exit 1
allure generate ./testreport/allure-results -c -o ./testreport/allure-report || exit 1
print_green "<--- 生成测试报告成功"
