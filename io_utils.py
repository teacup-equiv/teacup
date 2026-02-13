from typing import Tuple, Optional
import pandas as pd
import io


def read_table_from_bytes(content: bytes, filename: str = "uploaded_file", header_override: str = "Auto-detect") -> Tuple[pd.DataFrame, bool]:
    """
    Read a bytes payload (uploaded file) into a pandas DataFrame.
    Tries excel, then CSV with sep sniffing, then fallback parsing.
    Returns (df, header_used_bool).
    """
    fname = (filename or "").lower()
    ext = ""
    if "." in fname:
        ext = fname.split(".")[-1]

    # try reading with pandas robustly
    try:
        if ext in ("xls", "xlsx"):
            bio = io.BytesIO(content)
            raw = pd.read_excel(bio, header=None, dtype=object)
        else:
            bio = io.BytesIO(content)
            # try pandas csv sniffing with python engine sep=None
            raw = pd.read_csv(bio, header=None, dtype=object, sep=None, engine="python")
    except Exception:
        try:
            bio = io.BytesIO(content)
            raw = pd.read_csv(bio, header=None, dtype=object)
        except Exception:
            text = content.decode("utf-8", errors="replace")
            raw, _ = parse_table_from_text(text, header_override=header_override)  # fallback uses same logic

    raw = raw.replace({"": pd.NA, " ": pd.NA})

    # detect header usage and finalize columns (reuse shared logic)
    if header_override == "Force header" or header_override == "Force no header":
        # keep as-is and apply header override by re-running parse_table_from_text on text
        text = "\n".join(raw.astype(str).fillna("").apply(lambda row: ",".join(row.values), axis=1).tolist())
        df, header_used = parse_table_from_text(text, header_override=header_override)
        return df, header_used

    # default behavior: try to infer header
    # We already tried to read numeric types; call parse_table_from_text on textual representation
    text = "\n".join(raw.astype(str).fillna("").apply(lambda row: "\t".join(row.values), axis=1).tolist())
    df, header_used = parse_table_from_text(text, header_override="Auto-detect")
    return df, header_used


# -------------------------
# IO helper (paste text)
# -------------------------
def parse_table_from_text(raw_text: str, header_override: str = "Auto-detect") -> Tuple[pd.DataFrame, bool]:
    """
    Parse a pasted text table into a DataFrame.
    Attempts to detect delimiter (tabs or commas), infer header (unless forced),
    and returns (df, header_was_used_bool).
    """
    lines = [ln.rstrip() for ln in raw_text.splitlines()]
    lines = [ln for ln in lines if ln.strip() != ""]

    if len(lines) == 0:
        return pd.DataFrame(), False

    # detect delimiter
    if any("\t" in ln for ln in lines):
        delim_regex = r"[\t]+"
    elif any("," in ln for ln in lines):
        delim_regex = r"[,]+"
    else:
        delim_regex = None

    rows = []
    if delim_regex is None:
        for ln in lines:
            rows.append([ln.strip()])
    else:
        import re
        for ln in lines:
            parts = [p.strip() for p in re.split(delim_regex, ln)]
            rows.append(parts)

    max_cols = max(len(r) for r in rows)
    for r in rows:
        if len(r) < max_cols:
            r.extend([None] * (max_cols - len(r)))

    header_used = False
    if header_override == "Force header":
        header_used = True
    elif header_override == "Force no header":
        header_used = False
    else:
        # auto-detect header: first row non-numeric and later rows numeric-ish
        def is_numeric_token(tok):
            if tok is None:
                return False
            try:
                float(str(tok))
                return True
            except Exception:
                return False

        first = rows[0]
        later = rows[1:] if len(rows) > 1 else []
        first_numeric_counts = sum(1 for tok in first if is_numeric_token(tok))
        later_numeric_counts = 0
        for r in later:
            for tok in r:
                if is_numeric_token(tok):
                    later_numeric_counts += 1

        header_used = (first_numeric_counts == 0 and later_numeric_counts >= 1)

    if header_used:
        columns = [str(c) if c is not None else f"col{i}" for i, c in enumerate(rows[0], start=1)]
        data_rows = rows[1:]
        df = pd.DataFrame(data_rows, columns=columns)
    else:
        columns = [f"col{i}" for i in range(1, max_cols + 1)]
        df = pd.DataFrame(rows, columns=columns)

    # normalize empty strings to NA
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].replace({"": pd.NA, " ": pd.NA})

    return df, header_used