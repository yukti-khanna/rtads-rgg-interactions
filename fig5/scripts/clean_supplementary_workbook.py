"""Clean presentation text in XLSX exports without rewriting scientific cells.

Only XML text nodes are edited. Numeric cells, formulas, styles, worksheet
structure and all other workbook parts are preserved byte for byte.
"""
from pathlib import Path
import html
import re
import zipfile

HISTORY = re.compile(r'historic(?:ally|al)?[ _-]*', re.I)
FILE = re.compile(r'(?:/Users/|(?:[\w ./-]+\.(?:csv|xlsx|pkl|json|py|txt|md|joblib))\b)', re.I)
TEXT = re.compile(rb'(<(?:\w+:)?t(?:\s[^>]*)?>)(.*?)(</(?:\w+:)?t>)', re.S)

def clean_text(value):
    # Source paths and filenames are build details, not supplementary data.
    if FILE.search(value):
        # Drop path-only entries; keep scientific caveats in prose.
        lines = [line for line in value.splitlines() if not FILE.search(line)]
        value = '\n'.join(lines)
    value = re.sub(r'historical_source_column', 'feature_identifier', value, flags=re.I)
    value = re.sub(r'historical_alias', 'feature_alias', value, flags=re.I)
    value = HISTORY.sub('', value)
    if re.fullmatch(r'source[ _-]*(?:file|filename|path|workbook)s?', value.strip(), re.I):
        return ''
    return value

def clean_workbook(path):
    path = Path(path)
    temp = path.with_suffix('.cleaning.xlsx')
    edits = 0
    with zipfile.ZipFile(path) as src, zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_DEFLATED) as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename.endswith('.xml') and ('worksheets/' in item.filename or 'sharedStrings' in item.filename or 'comments' in item.filename):
                def replace(match):
                    nonlocal edits
                    original = html.unescape(match[2].decode('utf-8'))
                    cleaned = clean_text(original)
                    if cleaned == original:
                        return match[0]
                    edits += 1
                    return match[1] + html.escape(cleaned, quote=False).encode('utf-8') + match[3]
                data = TEXT.sub(replace, data)
            dst.writestr(item, data)
    if edits:
        temp.replace(path)
    else:
        temp.unlink()
    return edits

if __name__ == '__main__':
    import sys
    for arg in sys.argv[1:]:
        print(arg, clean_workbook(arg), 'text cells updated', flush=True)
