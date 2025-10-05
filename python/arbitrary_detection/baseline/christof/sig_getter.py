import requests
import pandas as pd
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='urllib3')

all_signatures = []
next_url = f"https://www.4byte.directory/api/v1/event-signatures/"

while next_url:
    try:
        resp = requests.get(next_url, timeout=10)
        if resp.status_code != 200:
            print(f"请求失败: {resp.status_code}, 等待20秒重试...")
            time.sleep(20)
            continue

        try:
            data = resp.json()
        except ValueError:
            print(f"响应不是 JSON，内容: {resp.text[:100]}...，等待20秒重试")
            time.sleep(20)
            continue

        for sig in data.get("results", []):
            all_signatures.append({
                'text_signature': sig.get("text_signature", ""),
                'hex_signature': sig.get("hex_signature", "")
            })

        next_url = data.get("next")
        print(f"已抓取 {len(all_signatures)} 条记录，下一页: {next_url}")

    except requests.exceptions.RequestException as e:
        print(f"请求异常: {e}, 等待20秒重试...")
        time.sleep(20)

# 去重并保存
df = pd.DataFrame(all_signatures)
df = df.drop_duplicates(subset=['hex_signature'], keep='first')
df.to_csv('event_signatures.csv', index=False)
print(f"已保存到 event_signatures.csv，共 {len(df)} 条去重后的记录")