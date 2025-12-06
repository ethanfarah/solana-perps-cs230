import re
import csv
from datetime import datetime

INPUT_FILE = "s&p.txt"
OUTPUT_FILE = "sp500_parsed.csv"

# Regex to capture each <tr>...</tr> block
TR_PATTERN = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL)
# Regex to capture each <td>...</td> value inside a row
TD_PATTERN = re.compile(r"<td[^>]*>(.*?)</td>")

def parse_line(html_row: str):
    # Extract raw cell values
    cells = TD_PATTERN.findall(html_row)
    if len(cells) < 7:
        return None  # skip malformed rows

    # Date
    raw_date = cells[0].strip()
    # Example format: "Nov 13, 2025"
    dt = datetime.strptime(raw_date, "%b %d, %Y")
    date_str = dt.strftime("%Y-%m-%d")

    # Helper to clean numeric strings like "5,473,720,000"
    def clean_num(s):
        return s.replace(",", "").strip()

    open_ = clean_num(cells[1])
    high = clean_num(cells[2])
    low = clean_num(cells[3])
    close = clean_num(cells[4])
    adj_close = clean_num(cells[5])
    # volume = clean_num(cells[6])  # available if you later want it

    return [date_str, open_, high, low, close, adj_close]

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    rows = []
    for tr_html in TR_PATTERN.findall(content):
        parsed = parse_line(tr_html)
        if parsed is not None:
            rows.append(parsed)

    # Sort by date ascending if you want
    rows.sort(key=lambda r: r[0])

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "open", "high", "low", "close", "adj_close"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()