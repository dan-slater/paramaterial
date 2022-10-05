"""Module for creating tables and figures for latex reports of paramaterial studies."""
import pandas as pd


def main():
    info_path = '../examples/baron study/info/01 baron info raw.xlsx'
    out_path = '../examples/baron study/output/01 baron info unique values.tex'
    make_unique_values_table(info_path=info_path, out_path=out_path)
    out_path = '../examples/baron study/output/01 baron info temperature rate.tex'
    make_temperature_rate_table(info_path=info_path, out_path=out_path)


def remove_characters_from_text_file(path: str, characters: list):
    """Remove characters from a text file."""
    with open(path, 'r') as file:
        text = file.read()
    for character in characters:
        text = text.replace(character, '')
    with open(path, 'w') as file:
        file.write(text)


def make_unique_values_table(info_path: str, out_path: str):
    """Create a latex table with the unique values for each column in the info table."""
    info_df = pd.read_excel(info_path)
    unique_info_df = pd.DataFrame(columns=['column', 'nr unique values', 'unique values'])
    for column in info_df.columns:
        unique_info_df = unique_info_df.append({'column': column,
                                                'nr unique values': info_df[column].nunique(),
                                                'unique values': info_df[column].unique()},
                                               ignore_index=True)
    unique_info_df.to_latex(out_path, index=False, escape=False)
    remove_characters_from_text_file(out_path, ['[', ']', ','])


def make_temperature_rate_table(info_path: str, out_path: str):
    """Create a latex table with the number of unique combinations of temperature and strain rate."""
    info_df = pd.read_excel(info_path)
    temperature_rate_df = pd.DataFrame(columns=['temperature', 'strain rate', 'nr combinations'])
    for temperature in info_df['temperature'].unique():
        for strain_rate in info_df['rate'].unique():
            temperature_rate_df = temperature_rate_df.append(
                {'temperature': temperature,
                 'strain rate': strain_rate,
                 'nr combinations':
                     len(info_df[(info_df['temperature'] == temperature)&(info_df['rate'] == strain_rate)])},
                ignore_index=True)
    temperature_rate_df.to_latex(out_path, index=False, escape=False)



if __name__ == '__main__':
    main()
