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
            print(info)
            if info.startswith("'data"):
                pass
            else:
                lines[i] = line.replace(f'DataSet({data}, {info})', f'DataSet({info}, {data})')
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


def main():
    """Run the main function."""
    swap_dataset_inputs('example/02 processing.ipynb')


if __name__ == '__main__':
    main()
