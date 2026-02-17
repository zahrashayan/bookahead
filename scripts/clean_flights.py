#!/usr/bin/env python3
import csv
import os
import re

INPUT = 'data/raw/flights_data_cleaned.csv'
OUTPUT_DIR = 'data/processed'
OUTPUT = os.path.join(OUTPUT_DIR, 'flights_data_cleaned_fixed.csv')

os.makedirs(OUTPUT_DIR, exist_ok=True)

NARROW_NBSP = '\u202F'

def fix_airline_text(txt):
    if not txt:
        return '', ''
    s = txt.strip()
    s = s.replace(NARROW_NBSP, ' ')
    # Insert separator when two tokens are concatenated (e.g. AlaskaHawaiian -> Alaska, Hawaiian)
    s = re.sub(r'([a-z])([A-Z])', r'\1, \2', s)
    # Extract "Operated by ..." into operated_by
    op = ''
    m = re.search(r'Operated by\s*(.*)', s, flags=re.IGNORECASE)
    if m:
        op = m.group(1).strip()
        # remove the operated part from airline string
        s = re.sub(r'Operated by\s*.*', '', s, flags=re.IGNORECASE).strip()
    # Normalize multiple separators and whitespace
    s = re.sub(r'\s+', ' ', s)
    s = s.strip(' ,')
    return s, op


def parse_line(line, header_len=12):
    # replace narrow no-break space
    line = line.replace(NARROW_NBSP, ' ')
    parts = line.rstrip('\n').split(',')
    if len(parts) < header_len:
        # pad
        parts += [''] * (header_len - len(parts))
    elif len(parts) > header_len:
        # Reconstruct: first 8 fields fixed, last 3 are departure_time, arrival_time, stops
        pre = parts[:8]
        post = parts[-3:]
        airline = ','.join(parts[8:-3])
        parts = pre + [airline] + post
    return parts


def extract_arrival_offset(arrival_raw):
    if not arrival_raw:
        return '', 0
    arrival_raw = arrival_raw.strip()
    m = re.match(r'^(.*?)(?:\+(\d+))?$', arrival_raw)
    if not m:
        return arrival_raw, 0
    time_part = m.group(1).strip()
    offset = int(m.group(2)) if m.group(2) else 0
    return time_part, offset


def main():
    processed = []
    processed_header = None
    skipped = 0
    with open(INPUT, 'r', encoding='utf-8') as inf:
        reader = inf
        header_line = next(reader)
        header = header_line.strip().split(',')
        # add new columns
        out_header = header + ['arrival_day_offset', 'operated_by']
        processed_header = out_header

        for line in reader:
            if not line.strip():
                continue
            parts = parse_line(line, header_len=len(header))
            if len(parts) != len(header):
                skipped += 1
                continue
            (scraped_date, scraped_time, route, departure_date,
             days_until_departure, day_of_week, is_weekend, price,
             airline_raw, departure_time_raw, arrival_time_raw, stops) = parts

            # clean airline and extract operated by
            airline, operated_by = fix_airline_text(airline_raw)

            # extract arrival offset
            arrival_time, arrival_offset = extract_arrival_offset(arrival_time_raw)

            # normalize times
            departure_time = departure_time_raw.strip()
            arrival_time = arrival_time.strip()

            # normalize price
            p = ''
            try:
                p = float(price) if price != '' else ''
            except Exception:
                p = ''

            out_row = [scraped_date, scraped_time, route, departure_date,
                       days_until_departure, day_of_week, is_weekend, p,
                       airline, departure_time, arrival_time, stops,
                       arrival_offset, operated_by]

            processed.append(out_row)

    # write cleaned CSV
    with open(OUTPUT, 'w', newline='', encoding='utf-8') as outf:
        writer = csv.writer(outf)
        writer.writerow(processed_header)
        for r in processed:
            writer.writerow(r)

    print(f'WROTE {len(processed)} rows to {OUTPUT} (skipped {skipped} malformed lines)')
    # print a small sample
    print('\nSAMPLE CLEANED (first 10 rows):')
    with open(OUTPUT, 'r', encoding='utf-8') as outf:
        for i, ln in enumerate(outf):
            print(ln.rstrip('\n'))
            if i >= 10:
                break


if __name__ == '__main__':
    main()
