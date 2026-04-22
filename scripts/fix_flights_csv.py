import csv
from collections import defaultdict
from pathlib import Path

INPATH = Path('data/raw/flights_data_cleaned.csv')
OUTPATH = Path('data/raw/flights_data_cleaned_fixed.csv')

def clean_field(s: str) -> str:
    if s is None:
        return ''
    s = s.strip()
    # normalize multiple double-quotes
    s = s.replace('"""', '"')
    s = s.replace('""', '"')
    # remove surrounding quotes
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]
    return s.strip()

def normalize_airline(a: str) -> str:
    a = clean_field(a)
    # common bad patterns -> canonical
    a = a.replace('Jet, Blue', 'JetBlue')
    a = a.replace('Jet,Blue', 'JetBlue')
    a = a.replace('Sky, West', 'SkyWest')
    a = a.replace('Sky,West', 'SkyWest')
    a = a.replace('SkyWest DBA', 'SkyWest DBA')
    a = a.replace('JetBlue', 'JetBlue')
    # collapse duplicated internal commas produced by bad quoting
    a = a.replace(' ,', ',')
    a = a.replace(', ', ',')
    return a

def main():
    with INPATH.open('r', encoding='utf-8') as f:
        header_line = f.readline().rstrip('\n')
        # parse header robustly
        try:
            header = next(csv.reader([header_line]))
        except Exception:
            header = header_line.split(',')
        header = [h.strip() for h in header]
        header_len = len(header)

        lines = f.read().splitlines()

    # scan sample to find max columns
    max_cols = header_len
    parsed_rows = []
    for i, raw in enumerate(lines, start=2):
        if not raw.strip():
            continue
        # skip exact repeated header lines
        if raw.strip() == header_line:
            continue
        try:
            row = next(csv.reader([raw]))
        except Exception:
            row = raw.split(',')
        parsed_rows.append((i, raw, row))
        if len(row) > max_cols:
            max_cols = len(row)

    # If an extra trailing column exists (data_source), add to header
    if max_cols > header_len:
        # append generic column names for extra columns if needed
        extras = max_cols - header_len
        for j in range(extras):
            if 'data_source' not in header:
                header.append('data_source')
            else:
                header.append(f'extra_{j}')

    # write cleaned rows to output
    counts = defaultdict(int)
    with OUTPATH.open('w', encoding='utf-8', newline='') as out:
        writer = csv.writer(out, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header)
        for i, raw, row in parsed_rows:
            counts['total'] += 1
            # normalize row length
            if len(row) < len(header):
                row = row + [''] * (len(header) - len(row))
            elif len(row) > len(header):
                # join extras into last column
                base = row[:len(header)-1]
                last = ','.join(row[len(header)-1:])
                row = base + [last]
            # clean fields
            row = [clean_field(c) for c in row]
            # normalize airlines and operated_by where applicable
            try:
                ai = header.index('airline')
                row[ai] = normalize_airline(row[ai])
            except ValueError:
                pass
            try:
                oi = header.index('operated_by')
                row[oi] = clean_field(row[oi])
            except ValueError:
                pass
            # skip spurious header rows that may appear (defensive)
            if row and row[0].lower() == 'scraped_date':
                counts['skipped_header_rows'] += 1
                continue
            writer.writerow(row)
            counts['written'] += 1

    # report basic stats
    print('input_header_len=', header_len)
    print('output_header_len=', len(header))
    print('rows_processed=', counts['total'])
    print('rows_written=', counts['written'])
    print('skipped_header_rows=', counts.get('skipped_header_rows', 0))

if __name__ == '__main__':
    main()
