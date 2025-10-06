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

failures=0

# 初始化XML文件内容
output_file="api_health_check.xml"
echo '<?xml version="1.0" encoding="UTF-8"?>' > "$output_file"
echo '<testsuites tests="4" failures="0" errors="0">' >> "$output_file"
echo '  <testsuite name="API Health Check" tests="4" failures="0" errors="0">' >> "$output_file"

print_green "---> SSE 订阅检测"

name="/stream"
REMOTE_SERVER="34.226.211.254"
URL="http://$REMOTE_SERVER:8545/stream?token=adminToken"
MAX_RETRIES=3
RETRY_COUNT=1
RESPONSE=""

while (( RETRY_COUNT <= MAX_RETRIES )); do
  print_yellow "尝试连接到 SSE 接口，尝试次数: $RETRY_COUNT"
  RESPONSE=$(curl -s --max-time 3 $URL)
  if echo "$RESPONSE" | grep -q "ping:"; then
    print_yellow "成功收到数据：$RESPONSE"
    break
  else
    print_yellow "未收到预期数据，重新尝试..."
    ((RETRY_COUNT++))
  fi
done
if (( RETRY_COUNT == MAX_RETRIES + 1 )); then
  print_red "SSE 订阅检测失败"
  (( failures++ ))
  echo "    <testcase name=\"$name\" status=\"failed\">" >> "$output_file"
  echo "      <failure>Response: $RESPONSE.</failure>" >> "$output_file"
  echo "    </testcase>" >> "$output_file"
else
  echo "    <testcase name=\"$name\" status=\"success\" />" >> "$output_file"
  print_green "<--- SSE 订阅检测成功"
fi

print_green "---> prometheus metrics 检测"

name="/debug/metrics/prometheus"
URL="http://$REMOTE_SERVER:6060/debug/metrics/prometheus"
MAX_RETRIES=3
RETRY_COUNT=1
RESPONSE=""

while (( RETRY_COUNT <= MAX_RETRIES )); do
  print_yellow "尝试获取 metrics，尝试次数: $RETRY_COUNT"
  RESPONSE=$(curl -s $URL)
  if echo "$RESPONSE" | grep -q "bundlepool_bundles"; then
    print_yellow "成功检测到：bundlepool_bundles 指标"
    break
  else
    print_yellow "未收到预期数据，重新尝试..."
    ((RETRY_COUNT++))
  fi
done
if (( RETRY_COUNT == MAX_RETRIES + 1 )); then
  print_red "prometheus metrics 检测失败"
  (( failures++ ))
  echo "    <testcase name=\"$name\" status=\"failed\">" >> "$output_file"
  echo "      <failure>Response: $RESPONSE.</failure>" >> "$output_file"
  echo "    </testcase>" >> "$output_file"
else
  echo "    <testcase name=\"$name\" status=\"success\" />" >> "$output_file"
  print_green "<--- prometheus metrics 检测成功"
fi

print_green "---> eth_sendRawTransaction api 检测"

name="/eth_sendRawTransaction"
URL="http://$REMOTE_SERVER:8545"
MAX_RETRIES=3
RETRY_COUNT=1
RESPONSE=""

while (( RETRY_COUNT <= MAX_RETRIES )); do
  print_yellow "尝试调用 eth_sendRawTransaction，尝试次数: $RETRY_COUNT"
  RESPONSE=$(curl -s -X POST $URL \
                  -H "Content-Type: application/json" \
                  -d '{
                      "jsonrpc": "2.0",
                      "method": "eth_sendRawTransaction",
                      "params": ["0xf868820442843b9aca008252089443dda9d1ac023bd3593dff5a1a677247bb98fe11822710808194a0b07ab9ed4d30245f116bc6e3c25a83ae7eb9d4dc69341dc5447e41c0303d4f76a05f7957ac12f9cea27d5d065b12f79bb1947d0f7861025b26488df97d060d6582"],
                      "id": 1
                  }'
  )
  if echo "$RESPONSE" | grep -q "nonce too low"; then
    print_yellow "成功收到接口返回数据 $RESPONSE"
    break
  else
    print_yellow "未收到预期数据，重新尝试..."
    ((RETRY_COUNT++))
  fi
done
if (( RETRY_COUNT == MAX_RETRIES + 1 )); then
  print_red "eth_sendRawTransaction api 检测失败"
  (( failures++ ))
  echo "    <testcase name=\"$name\" status=\"failed\">" >> "$output_file"
  echo "      <failure>Response: $RESPONSE.</failure>" >> "$output_file"
  echo "    </testcase>" >> "$output_file"
else
  echo "    <testcase name=\"$name\" status=\"success\" />" >> "$output_file"
  print_green "<--- eth_sendRawTransaction api 检测成功"
fi

print_green "---> eth_sendMevBundle api 检测"

name="/eth_sendMevBundle"
URL="http://$REMOTE_SERVER:8545"
MAX_RETRIES=3
RETRY_COUNT=1
RESPONSE=""

while (( RETRY_COUNT <= MAX_RETRIES )); do
  print_yellow "尝试调用 eth_sendMevBundle，尝试次数: $RETRY_COUNT"
  RESPONSE=$(curl -s -X POST $URL \
                  -H "Content-Type: application/json" \
                  -d '{
                      "jsonrpc": "2.0",
                      "method": "eth_sendMevBundle",
                      "params": [{
                        "txs": ["0xf868820442843b9aca008252089443dda9d1ac023bd3593dff5a1a677247bb98fe11822710808194a0b07ab9ed4d30245f116bc6e3c25a83ae7eb9d4dc69341dc5447e41c0303d4f76a05f7957ac12f9cea27d5d065b12f79bb1947d0f7861025b26488df97d060d6582"]
                      }],
                      "id": 1
                  }'
  )
  if echo "$RESPONSE" | grep -q "nonce too low"; then
    print_yellow "成功收到接口返回数据 $RESPONSE"
    break
  else
    print_yellow "未收到预期数据，重新尝试..."
    ((RETRY_COUNT++))
  fi
done
if (( RETRY_COUNT == MAX_RETRIES + 1 )); then
  print_red "eth_sendMevBundle api 检测失败"
  (( failures++ ))
  echo "    <testcase name=\"$name\" status=\"failed\">" >> "$output_file"
  echo "      <failure>Response: $RESPONSE.</failure>" >> "$output_file"
  echo "    </testcase>" >> "$output_file"
else
  echo "    <testcase name=\"$name\" status=\"success\" />" >> "$output_file"
  print_green "<--- eth_sendMevBundle api 检测成功"
fi

# 更新失败计数
if [[ "$(uname)" == "Darwin" ]]; then
  sed -i "" "s/failures=\"0\"/failures=\"$failures\"/g" "$output_file"
else
  sed -i "s/failures=\"0\"/failures=\"$failures\"/g" "$output_file"
fi
# 结束XML文件
echo '  </testsuite>' >> "$output_file"
echo '</testsuites>' >> "$output_file"

print_green "API health check XML file generated: $output_file"
