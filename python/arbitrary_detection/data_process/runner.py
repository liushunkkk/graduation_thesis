import os

import numpy as np
import pandas as pd
import pymysql

database_config = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "root123456",
    "database": "arbitrary",
    "charset": "utf8mb4"
}

common_sql = """
    SELECT
      -- 共有字段（只取 transactions 表的版本）
      t.tx_hash,
      t.tx_type,
      t.block_number,

      -- ethereum_transactions 独有字段
      t.nonce,
      t.gas_price,
      t.gas_tip_cap,
      t.gas_fee_cap,
      t.gas,
      t.to,
      t.value,
      t.data,
      t.access_list,
      t.v,
      t.r,
      t.s,
      t.origin_json_string AS transaction_json,

      -- ethereum_receipts 独有字段
      r.post_state,
      r.status,
      r.cumulative_gas_used,
      r.bloom,
      r.logs,
      r.contract_address,
      r.gas_used,
      r.effective_gas_price,
      r.blob_gas_used,
      r.blob_gas_price,
      r.block_hash,
      r.transaction_index,
      r.origin_json_string AS receipt_json
    FROM
      {}_transactions t
    INNER JOIN
      {}_receipts r ON t.tx_hash = r.tx_hash
    LIMIT %s;
"""


class DataProcessRunner:
    def __init__(self):
        self.name = "DataProcessRunner"
        self.TableEthereum = "ethereum"
        self.TableComparison = "comparison"

    def get_rows(self, table, limit):
        """
        获取所有的数据
        Table 表类型，传 self.TableEthereum 或 self.TableComparison: positive or negative
        limit 条数
        """
        conn = pymysql.connect(**database_config)
        with conn.cursor() as cursor:
            cursor.execute(common_sql.format(table, table), (limit,))
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
        return columns, rows

    def transfer_to_dataframe(self, columns, rows):
        """将数据库查出的数据转换为dataframe结构"""
        return pd.DataFrame(rows, columns=columns)

    def filter_rows_within_data_length(self, df):
        """使用分位截断根据data长度过滤数据"""
        data_list = df['data'].tolist()
        data_len_list = [len(data) for data in data_list]
        lower = np.percentile(data_len_list, 0.1 * 100)
        upper = np.percentile(data_len_list, 0.9 * 100)
        df = df[df['data'].str.len().between(lower, upper)]
        return lower, upper, df

    def output_tx_hash(self, df, table):
        """
        输出tx_hash到文件中
        :param df: 数据
        :param table: 表类型
        """
        if table == self.TableEthereum:
            file_name = "../files/positive_tx_hashes.csv"
        else:
            file_name = "../files/negative_tx_hashes.csv"
        if os.path.exists(file_name):
            os.remove(file_name)
        df[['tx_hash']].to_csv(file_name, index=False)

    def output_all(self, df, table):
        """
        输出全部数据
        :param df: 数据
        :param table: 表类型
        """
        if table == self.TableEthereum:
            file_name = '../files/positive_data.csv'
        else:
            file_name = '../files/negative_data.csv'
        if os.path.exists(file_name):
            os.remove(file_name)
        df.to_csv(file_name, index=False)


def operate_positive(limit=10000):
    r = DataProcessRunner()
    columns, rows = r.get_rows(r.TableEthereum, limit)
    print(columns)

    df = r.transfer_to_dataframe(columns, rows)
    print(df)

    # low, upper, df = r.filter_rows_within_data_length(df)
    # print(low, upper)

    print(df)

    r.output_tx_hash(df, r.TableEthereum)

    r.output_all(df, r.TableEthereum)


def operate_negative(limit=10000):
    r = DataProcessRunner()
    columns, rows = r.get_rows(r.TableComparison, limit)
    print(columns)

    df = r.transfer_to_dataframe(columns, rows)
    print(df)

    # low, upper, df = r.filter_rows_within_data_length(df)
    # print(low, upper)

    print(df)

    r.output_tx_hash(df, r.TableComparison)

    r.output_all(df, r.TableComparison)


if __name__ == '__main__':
    operate_positive(100000)
    operate_negative(200000)
