import csv
import os
import warnings
import uuid
import base64

# Clean a row by encoding the name
def clean_row(row):
    row['name'] = base64.b64encode(row['name'].encode()).decode() if 'name' in row else row['name']
    return row

# Clear a folder of all files
def clear_folder(folder: str) -> None:
    for file in os.listdir(folder):
        try:
            os.remove(os.path.join(folder, file))
        except Exception as e:
            warnings.warn(f"[datascrape -> clear_folder()] Could not remove {file}: {e}")

# Compile a demo into a CSV
def comp_into_csv(demoname: str) -> None:
    os.system(f'cd data_collector && cargo run --release -- -i "../demos/{demoname}" -a viewangles_to_csv')

# Filter the CSV by one player
def filter_by_one_player(data: str) -> None:
    with open(data, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        first_player = next(reader).get('name')
        filtered_rows = [clean_row(row) for row in reader if row.get('name') == first_player]

        with open(data, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(filtered_rows)

# Concatenate all CSVs in a folder into one CSV
def concatenate_csvs(data_folder: str, name='data.csv') -> None:
    csvs = os.listdir(data_folder)
    seen_ticks = set()
    with open(name, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['tick', 'name', 'steam_id', 'origin_x', 'origin_y', 'origin_z', 'viewangle', 'pitchangle', 'va_delta', 'pa_delta'])
        for csv_file in csvs:
            with open(data_folder + "/" + csv_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    tick_name_pair = (row[0], row[1])
                    if tick_name_pair not in seen_ticks:
                        seen_ticks.add(tick_name_pair)
                        writer.writerow(row)