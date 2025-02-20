import csv
import os
import warnings

def clear_folder(folder) -> None:
    for file in os.listdir(folder):
        try: 
            os.remove(f"{folder}/{file}")

        except: warnings.warn(f"[datascrape -> clear_folder()] Could not remove {str(file)}")

def filter_by_name(data, name) -> list:
    with open(data, 'r') as file:
        reader = csv.DictReader(file)
        filtered_data = [row for row in reader if row['name'] == name]
    return filtered_data

def filter_by_ticks(data, start_t, end_t) -> list:
    with open(data, 'r') as file:
        reader = csv.DictReader(file)
        filtered_data = [row for row in reader if row['tick'] >= start_t and row['tick'] <= end_t]
    return filtered_data

def comp_into_csv(demoname) -> None:
    os.system(f'cd ./data_collector; cargo run --release -- -i "../demos/{demoname}" -a viewangles_to_csv')

def comp_all_csv() -> bool:
    demos = os.listdir('./demos')
    for demo in demos:
        print(f"Compiling {demo} into CSV")
        comp_into_csv(demo)
    return True