"""Script for refactoring a Python file."""


def swap_dataset_inputs(path: str):
    """Swap the order of the inputs to DataSet.

    Args:
        path (str): Path to the file to refactor.
    """
    with open(path, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if 'DataSet(' in line:
            data = line.split('DataSet(')[1].split(',')[0]
            info = line.split('DataSet(')[1].split(', ')[1].split(')')[0]
            if info.startswith("'data"):
                print(info)
            else:
                lines[i] = line.replace(f'DataSet({data}, {info}', f'DataSet({info}, {data}')
                print(f'Line {i} changed from {line} to {lines[i]}')
        if 'write_output(' in line:
            data = line.split('write_output(')[1].split(',')[0]
            info = line.split('write_output(')[1].split(', ')[1].split(')')[0]
            print(info)
            if info.startswith("'data"):
                pass
            else:
                lines[i] = line.replace(f'write_output({data}, {info})', f'write_output({info}, {data})')
                print(f'Line {i} changed from {line} to {lines[i]}')

    with open(path, 'w') as f:
        f.writelines(lines)


def command_line_main():
    import argparse
    parser = argparse.ArgumentParser(description='Refactor python file to swap data and info arguments.')
    parser.add_argument("file_path", help="Path to the file to refactor.")
    args = parser.parse_args()
    swap_dataset_inputs(args.file_path)


if __name__ == '__main__':
    command_line_main()
    # main()
