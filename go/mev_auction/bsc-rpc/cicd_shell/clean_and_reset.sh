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

print_green "---> 恢复原始geth"
ssh "root@$REMOTE_SERVER" <<'EOF'
if [ -f /root/cicd/old-geth/geth ]; then
    echo "File /root/cicd/old-geth/geth exists. Proceeding with operations."
    mv -f /root/cicd/old-geth/geth /root/bsc-rpc/cmd/geth/geth
else
    echo "File /root/cicd/old-geth/geth does not exist. "
    exit 1
fi
EOF
if [ $? -ne 0 ]; then
  print_red "<--- 原始geth不存在，需要重新拉取feat/rpc分支最新代码构建"
else
  print_green "<--- 恢复原始geth成功"
fi

print_green "---> 恢复geth节点"
ssh "root@$REMOTE_SERVER" <<'EOF'
function check_and_run_geth() {
  COUNT=1
  MAX_RETRIES=3
  while (( COUNT <= MAX_RETRIES )); do
    echo "Attempt $COUNT to start geth..."
    /root/geth_for_cicd.sh restart bsc-dev
    STATUS=$?
    if [[ $STATUS -eq 0 ]]; then
      echo "Geth started successfully!"
      exit 0
    else
      echo "Geth failed to start. Exit code: $STATUS"
    fi
    (( COUNT++ ))
  done
  echo "Geth failed to start after $MAX_RETRIES attempts."
  exit 1
}
check_and_run_geth
EOF
if [ $? -ne 0 ]; then
  print_red "<--- 恢复geth节点失败"
  exit 1
else
  print_green "<--- 恢复geth节点成功"
fi
