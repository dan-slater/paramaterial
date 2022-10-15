"""Module for examnple study of baron data."""
from paramaterial.plug import DataSet


def main():
    """Main function."""
    dataset = DataSet(data_dir='data/01 prepared data', info_path='info/01 prepared info.xlsx')

    info_table = dataset.info_table

    value_counts_multi_indexed_series = info_table[['material', 'temperature', 'rate']].value_counts().sort_index()

    value_counts_multi_indexed_df = value_counts_multi_indexed_series.to_frame().reset_index()
    value_counts_multi_indexed_df.columns = ['material', 'temperature', 'rate', 'count']
    value_counts_multi_indexed_df = value_counts_multi_indexed_df.pivot_table(index=['material', 'rate'], columns='temperature', values='count')
    value_counts_multi_indexed_df = value_counts_multi_indexed_df.fillna(0).astype(int)
    print(value_counts_multi_indexed_df)



if __name__ == '__main__':
    main()