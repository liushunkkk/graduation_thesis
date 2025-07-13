from data_process import runner


def operate_positive():
    r = runner.DataProcessRunner()
    columns, rows = r.get_rows(r.TableEthereum, 10000)
    print(columns)

    df = r.transfer_to_dataframe(columns, rows)
    print(df)

    low, upper, df = r.filter_rows_within_data_length(df)
    print(low, upper)

    print(df)

    r.output_tx_hash(df, r.TableEthereum)

    r.output_all(df, r.TableEthereum)


def operate_negative():
    r = runner.DataProcessRunner()
    columns, rows = r.get_rows(r.TableComparison, 20000)
    print(columns)

    df = r.transfer_to_dataframe(columns, rows)
    print(df)

    low, upper, df = r.filter_rows_within_data_length(df)
    print(low, upper)

    print(df)

    r.output_tx_hash(df, r.TableComparison)

    r.output_all(df, r.TableComparison)


if __name__ == '__main__':
    operate_positive()
    operate_negative()
