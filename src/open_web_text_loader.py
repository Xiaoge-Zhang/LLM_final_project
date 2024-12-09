import os
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.csv
import pyarrow.ipc


def arrow_files_in_dir(directory):
    files = []
    for filename in os.listdir(directory):
        if filename.endswith('.arrow') and os.path.isfile(os.path.join(directory, filename)):
            files.append(filename)
    return files


if __name__ == '__main__':
    raw_data_dir = '../data/train/'
    vocab_file = 'open_web_vocab.txt'
    train_file_name = 'open_web_train.txt'
    test_file_name = 'open_web_test.txt'

    files = arrow_files_in_dir(raw_data_dir)

    num_files = len(files)
    train_test_split = 0.9

    train_size = int(num_files * train_test_split)
    train_files = files[:train_size]
    test_files = files[train_size:]

    vocab = set()

    # save the traning files
    with open('../processed_data/' + train_file_name, 'w', encoding='utf-8') as output_f:
        for input_i, filename in enumerate(tqdm(train_files, total=len(train_files))):
            file_dir = os.path.join(raw_data_dir, filename)
            with pa.memory_map(file_dir, "r") as arrow_f:
                reader = pa.ipc.RecordBatchStreamReader(arrow_f)
                table = reader.read_all()
            # Write row data
            for row in table.to_pydict().values():
                for record in zip(*table.to_pydict().values()):
                    text = record[0]
                    output_f.write(text)
                    characters = set(text)
                    vocab.update(characters)


    # save the test files
    with open('../processed_data/' + test_file_name, 'w', encoding='utf-8') as output_f:
        for input_i, filename in enumerate(tqdm(test_files, total=len(test_files))):
            file_dir = os.path.join(raw_data_dir, filename)
            with pa.memory_map(file_dir, "r") as arrow_f:
                reader = pa.ipc.RecordBatchStreamReader(arrow_f)
                table = reader.read_all()
            # Write row data
            for row in table.to_pydict().values():
                for record in zip(*table.to_pydict().values()):
                    text = record[0]
                    output_f.write(text)
                    characters = set(text)
                    vocab.update(characters)


    # save the vocab
    with open('../processed_data/' + vocab_file, 'w', encoding='utf-8') as v_f:
        for char in sorted(list(vocab)):
            v_f.write(char + '\n')