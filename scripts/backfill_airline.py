import csv
from pathlib import Path
from collections import defaultdict

IN = Path('data/raw/flights_data_cleaned.csv')

def backfill(rows, header):
    changed = []
    n = len(rows)
    for i, r in enumerate(rows):
        if r.get('airline','').strip():
            continue
        route = r.get('route','').strip()
        dep = r.get('departure_date','').strip()
        filled = None
        # search backward for same route & departure_date
        j = i-1
        while j >= 0:
            rr = rows[j]
            if rr.get('route','').strip() == route and rr.get('departure_date','').strip() == dep and rr.get('airline','').strip():
                filled = rr['airline']
                break
            j -= 1
        # search forward for same route & departure_date
        if not filled:
            j = i+1
            while j < n:
                rr = rows[j]
                if rr.get('route','').strip() == route and rr.get('departure_date','').strip() == dep and rr.get('airline','').strip():
                    filled = rr['airline']
                    break
                j += 1
        # backward same route
        if not filled:
            j = i-1
            while j >= 0:
                rr = rows[j]
                if rr.get('route','').strip() == route and rr.get('airline','').strip():
                    filled = rr['airline']
                    break
                j -= 1
        # forward same route
        if not filled:
            j = i+1
            while j < n:
                rr = rows[j]
                if rr.get('route','').strip() == route and rr.get('airline','').strip():
                    filled = rr['airline']
                    break
                j += 1
        # fallback: previous non-empty airline anywhere
        if not filled:
            j = i-1
            while j >= 0:
                rr = rows[j]
                if rr.get('airline','').strip():
                    filled = rr['airline']
                    break
                j -= 1
        if filled:
            rows[i]['airline'] = filled
            changed.append({'line': i+2, 'filled_with': filled})
    return changed

def main():
    with IN.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)

    missing_before = sum(1 for r in rows if not r.get('airline','').strip())
    changed = backfill(rows, header)
    missing_after = sum(1 for r in rows if not r.get('airline','').strip())

    # write back
    tmp = IN.with_suffix('.tmp')
    with tmp.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    IN.unlink()
    tmp.rename(IN)

    print('missing_before=', missing_before)
    print('changed_count=', len(changed))
    print('missing_after=', missing_after)
    if len(changed) <= 20:
        for c in changed:
            print(c)

if __name__ == '__main__':
    main()
