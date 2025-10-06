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

print_green "----> 二进制打包阶段"

print_yellow "当前目录：$(pwd)"

print_yellow "进入项目目录"
cd ../ && pwd && cd clone_project/bsc-rpc  && pwd || exit 1

print_yellow "当前分支: $(git rev-parse --abbrev-ref HEAD)"
print_yellow "当前提交ID：$(git log -n 10 --pretty=format:"%H")"

print_yellow "Go env"
export GO111MODULE=on
go env || exit 1
print_yellow "make geth..."
make geth || exit 1
print_green "<---- 二进制文件编译成功"

print_green "---> 替换原始geth"
ssh "root@$REMOTE_SERVER" <<'EOF'
mv -f /var/lib/docker/volumes/runner-t3sztqn-project-60522421-concurrent-0-cache-c33bcaa1fd2c77edfc3893b41966cea8/_data/shengweifeng/clone_project/bsc-rpc/build/bin/geth  /root/cicd || exit 1
ls -l /root/cicd
rm -f /root/cicd/old-geth/geth && cp /root/bsc-rpc/cmd/geth/geth /root/cicd/old-geth && mv -f /root/cicd/geth /root/bsc-rpc/cmd/geth/geth
EOF
if [ $? -ne 0 ]; then
  print_red "<--- 替换原始geth失败"
  exit 1
else
  print_green "<--- 替换原始geth成功"
fi

print_green "---> 处理启动脚本"
ssh "root@$REMOTE_SERVER" <<'EOF'
if [ -f '/root/geth_for_cicd.sh' ]; then echo '文件存在，授权执行权限'; else echo '文件不存在，退出'; exit 1; fi
chmod +x /root/geth_for_cicd.sh
EOF
if [ $? -ne 0 ]; then
  print_red "<--- 处理启动脚本失败"
  exit 1
else
  print_green "<--- 处理启动脚本成功"
fi

print_green "---> 启动程序"
ssh "root@$REMOTE_SERVER" <<'EOF'
function check_and_run_geth() {
  COUNT=1        # 当前尝试次数
  MAX_RETRIES=3  # 最大重试次数

  while (( COUNT <= MAX_RETRIES )); do
    echo "Attempt $COUNT to start geth..."
    # 执行启动命令
    /root/geth_for_cicd.sh restart bsc-dev
    STATUS=$?  # 获取命令退出状态码

    if [[ $STATUS -eq 0 ]]; then
      echo "Geth started successfully!"
      exit 0
    else
      echo "Geth failed to start. Exit code: $STATUS"
    fi
    (( COUNT++ ))  # 增加尝试次数
  done
  echo "Geth failed to start after $MAX_RETRIES attempts."
  exit 1
}
check_and_run_geth
EOF
if [ $? -ne 0 ]; then
  print_red "<--- 启动程序失败"
  exit 1
else
  print_green "<--- 启动程序成功"
fi
