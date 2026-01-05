import pandas as pd

xlsx = pd.ExcelFile("odchylka.xlsx")
for sheet in xlsx.sheet_names:
    df = pd.read_excel("odchylka.xlsx", sheet_name=sheet)
    df.to_csv(f"{sheet}.csv", index=False, encoding="utf-8")

xlsx = pd.ExcelFile("regulacna.xlsx")
for sheet in xlsx.sheet_names:
    df = pd.read_excel("regulacna.xlsx", sheet_name=sheet)
    df.to_csv(f"{sheet}.csv", index=False, encoding="utf-8")