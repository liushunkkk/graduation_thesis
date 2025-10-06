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

print_yellow "检测python环境"
python3 --version || exit 1
pip --version || exit 1

print_yellow "当前目录：$(pwd)"

print_green "---> 创建系统测试目录"
cd ../ && pwd && rm -rf blockrazor_system_test && mkdir blockrazor_system_test && cd blockrazor_system_test || exit 1
pwd
print_green "<--- 创建系统测试成功"

print_green "---> 克隆系统测试代码"
git clone --branch feat-rpc-test https://gitlab.com/Momo_Li/scutum_test.git || exit 1
print_green "---> 克隆系统测试代码成功"


print_green "---> 执行系统测试"
cd scutum_test || exit 1
pwd
print_yellow "验证 script/system_test.sh 是否存在..."
if [ ! -f "script/system_test.sh" ]; then
  print_red "执行脚本不存在";
  exit 1;
fi
print_yellow "授权执行权限"
chmod +x script/system_test.sh || exit 1
print_yellow "执行脚本"
export ChainID=56
export BRANCH=""
cd script || exit 1
pwd
./system_test.sh || exit 1
print_green "<--- 执行系统测试成功"


