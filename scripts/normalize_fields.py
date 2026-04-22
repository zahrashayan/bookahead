import csv
import re
from pathlib import Path
from collections import defaultdict

INPATH = Path('data/raw/flights_data_cleaned.csv')
OUTPATH = Path('data/raw/flights_data_cleaned_norm.csv')

def normalize_airline(a: str) -> str:
    if a is None:
        return ''
    s = a.strip()
    # remove all double-quote characters
    s = s.replace('"', '')
    # collapse whitespace
    s = re.sub(r'\s+', ' ', s)
    # normalize common mis-splits
    s = s.replace('Jet, Blue', 'JetBlue').replace('Jet,Blue', 'JetBlue')
    s = s.replace('JetBlue', 'JetBlue')
    s = s.replace('Sky, West', 'SkyWest').replace('Sky,West', 'SkyWest')
    # remove stray leading/trailing commas/spaces
    s = s.strip(' ,')
    # collapse internal comma spacing
    s = re.sub(r'\s*,\s*', ',', s)
    return s

def normalize_operated_by(o: str) -> str:
    if o is None:
        return ''
    s = o.strip()
    s = s.replace('"', '')
    s = re.sub(r'\s+', ' ', s)
    s = s.strip(' ,')
    return s

def main():
    counts = defaultdict(int)
    with INPATH.open('r', encoding='utf-8') as inf:
        reader = csv.DictReader(inf)
        fieldnames = reader.fieldnames
        rows = list(reader)

    for r in rows:
        counts['total'] += 1
        if 'airline' in r:
            orig = r['airline']
            new = normalize_airline(orig)
            if new != orig:
                counts['airline_changed'] += 1
            r['airline'] = new
        if 'operated_by' in r:
            orig = r['operated_by']
            new = normalize_operated_by(orig)
            if new != orig:
                counts['operated_by_changed'] += 1
            r['operated_by'] = new

    with OUTPATH.open('w', encoding='utf-8', newline='') as outf:
        writer = csv.DictWriter(outf, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # replace input file atomically
    INPATH.unlink()
    OUTPATH.rename(INPATH)

    print('rows=', counts['total'])
    print('airline_changed=', counts.get('airline_changed', 0))
    print('operated_by_changed=', counts.get('operated_by_changed', 0))

if __name__ == '__main__':
    main()
