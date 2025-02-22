import csv
import os
import warnings
import re
import base64

def clean_row(row):
    row['name'] = base64.b64encode(row['name'].encode()).decode() if 'name' in row else row['name']
    return row

def clear_folder(folder: str) -> None:
    for file in os.listdir(folder):
        try:
            os.remove(os.path.join(folder, file))
        except Exception as e:
            warnings.warn(f"[datascrape -> clear_folder()] Could not remove {file}: {e}")

def filter_by_name(data: str, name: str) -> list:
    with open(data, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        return [row for row in reader if row.get('name') == name]

def filter_by_ticks(data: str, start_t: int, end_t: int) -> list:
    with open(data, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        return [row for row in reader if start_t <= int(row.get('tick', 0)) <= end_t]

def comp_into_csv(demoname: str) -> None:
    os.system(f'cd data_collector && cargo run --release -- -i "../demos/{demoname}" -a viewangles_to_csv')

def comp_all_csv() -> bool:
    demos = os.listdir('demos')
    for demo in demos:
        print(f"Compiling {demo} into CSV")
        comp_into_csv(demo)
    return True

def filter_by_one_player(data: str) -> None:
    with open(data, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        first_player = next(reader).get('name')
        filtered_rows = [clean_row(row) for row in reader if row.get('name') == first_player]

        with open(data, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(filtered_rows)

def concatenate_csvs(data_folder: str) -> None:
    csvs = os.listdir(data_folder)
    with open('data.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['tick', 'name', 'steam_id', 'origin_x', 'origin_y', 'origin_z', 'viewangle', 'pitchangle', 'va_delta', 'pa_delta'])
        for csv_file in csvs:
            with open(os.path.join(data_folder, csv_file), 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    writer.writerow(row)

