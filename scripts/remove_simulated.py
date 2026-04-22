import csv
from pathlib import Path

INPATH = Path('data/raw/flights_data_cleaned.csv')
TMPPATH = Path('data/raw/flights_data_cleaned_tmp.csv')

def main():
    with INPATH.open('r', encoding='utf-8') as inf, TMPPATH.open('w', encoding='utf-8', newline='') as outf:
        reader = csv.DictReader(inf)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise SystemExit('no header found')
        writer = csv.DictWriter(outf, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        total = 0
        changed = 0
        for r in reader:
            total += 1
            ds = r.get('data_source', '')
            if ds is not None and 'simulated' in ds.lower():
                r['data_source'] = ''
                changed += 1
            writer.writerow(r)

    # replace original file
    INPATH.unlink()
    TMPPATH.rename(INPATH)
    print(f'total={total} changed={changed}')

if __name__ == '__main__':
    main()
