import pandas as pd

# xlsx = pd.ExcelFile("odchylka.xlsx")
# for sheet in xlsx.sheet_names:
#     df = pd.read_excel("odchylka.xlsx", sheet_name=sheet)
#     df.to_csv(f"{sheet}.csv", index=False, encoding="utf-8")

xlsx = pd.ExcelFile("regulacna_jan_2026.xlsx")
# for sheet in xlsx.sheet_names:
#     df = pd.read_excel("regulacna_jan_2026.xlsx", sheet_name=sheet)
#     df.to_csv(f"januar/{sheet}.csv", index=False, encoding="utf-8")

df = pd.read_excel("regulacna_jan_2026.xlsx", sheet_name="aFRR")
raw = df
print(type(df.columns))
print(df.columns)

def norm(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()

# 2) nájdi riadok, kde sa objaví "objem" aj "štandard" (hlavička)
header_row = None
for i in range(min(80, len(raw))):
    row_text = " | ".join(norm(v) for v in raw.iloc[i].tolist())
    if ("objem" in row_text) and ("štandard" in row_text or "standard" in row_text) and ("mwh" in row_text):
        header_row = i
        break

if header_row is None:
    raise RuntimeError("Nenašiel som riadok hlavičky (objem/štandardná cena). Skús zvýšiť rozsah alebo skontroluj XLSX.")

# 3) často je nad tým ešte riadok s UP/DOWN -> skúsme ho zobrať
updown_row = header_row - 1 if header_row > 0 else header_row

# forward-fill pre merged cells (UP môže byť iba v prvom stĺpci bloku)
updown = raw.iloc[updown_row].copy().ffill()
subhdr = raw.iloc[header_row].copy()

# 4) zlož finálne názvy stĺpcov
cols = []
for a, b in zip(updown.tolist(), subhdr.tolist()):
    name = (str(a).strip() if not pd.isna(a) else "") + "|" + (str(b).strip() if not pd.isna(b) else "")
    cols.append(name.lower())

data = raw.iloc[header_row + 1:].copy()
data.columns = cols

# 5) vyber 4 stĺpce "up objem", "up cena", "down objem", "down cena"
def find_col(must_have):
    for c in data.columns:
        if all(k in c for k in must_have):
            return c
    raise KeyError(f"Nenašiel som stĺpec pre {must_have}. Mám stĺpce: {data.columns.tolist()}")

c_up_objem   = find_col(["up", "objem"])
c_up_cena    = find_col(["up", "cena"])
c_down_objem = find_col(["down", "objem"])
c_down_cena  = find_col(["down", "cena"])

df2 = data[[c_up_objem, c_up_cena, c_down_objem, c_down_cena]].copy()
df2.columns = ["Up_Objem", "Up_Cena", "Down_Objem", "Down_Cena"]

# 6) vyhoď prázdne riadky (kde nie sú čísla)
for c in df2.columns:
    df2[c] = pd.to_numeric(df2[c], errors="coerce")

df2 = df2.dropna(how="all")

df2.to_csv(out_csv, index=False, encoding="utf-8")
print(f"OK -> {out_csv}")