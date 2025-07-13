import pymysql

# 源数据库配置
source_config = {
    "host": "127.0.0.1",
    "port": 3307,
    "user": "atomic_user",
    "password": "atomic_user",
    "database": "bscreward",
    "charset": "utf8mb4"
}

# 目标数据库配置
target_config = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "root123456",
    "database": "arbitrary",
    "charset": "utf8mb4"
}


# ======== SQL 模板（无排序） ========
select_sql = """
SELECT searcher, builder, `from`, `to`, block_num, tx_hash, time_stamp, mev_type, 
       position, bribe_value, bribee, bribe_type, arb_profit 
FROM searcher_builders 
WHERE mev_type = 'atomic' 
LIMIT %s OFFSET %s
"""

insert_sql = """
INSERT IGNORE INTO arbitrary_transaction 
(searcher, builder, `from`, `to`, block_num, tx_hash, time_stamp, mev_type, 
 position, bribe_value, bribee, bribe_type, arb_profit)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

# ======== 分页参数 ========
batch_size = 5000
offset = 0

# ======== 开始迁移 ========
try:
    source_conn = pymysql.connect(**source_config)
    target_conn = pymysql.connect(**target_config)

    while True:
        with source_conn.cursor() as source_cursor, target_conn.cursor() as target_cursor:
            source_cursor.execute(select_sql, (batch_size, offset))
            rows = source_cursor.fetchall()

            if not rows:
                print("✅ 数据迁移完成。")
                break

            target_cursor.executemany(insert_sql, rows)
            target_conn.commit()

            print(f"✅ 已迁移 {len(rows)} 条数据（offset = {offset}）")

        offset += batch_size

finally:
    source_conn.close()
    target_conn.close()