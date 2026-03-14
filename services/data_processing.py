import pandas as pd
import numpy as np
import io

def read_uploaded_file(file):
    filename = (file.filename or "").lower()

    if filename.endswith(".csv"):
        try:
            return pd.read_csv(file, encoding="utf-8")
        except UnicodeDecodeError:
            file.seek(0)
            return pd.read_csv(file, encoding="latin1")

    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        return pd.read_excel(file)

    raise ValueError("Please upload a CSV or Excel file (.csv, .xlsx, .xls)")

def try_parse_dates(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            sample = df[col].dropna().astype(str).head(20)
            if sample.empty:
                continue

            date_hint = any(
                x in col.lower()
                for x in ["date", "time", "month", "year", "created", "updated", "start", "end", "joined", "dob"]
            )

            if date_hint:
                try:
                    parsed = pd.to_datetime(df[col], errors="coerce")
                    if parsed.notna().sum() >= max(1, len(df) // 5):
                        df[col] = parsed
                except Exception:
                    pass
    return df

def smart_detect(df):
    cl = {c: c.lower().replace(" ", "").replace("_", "") for c in df.columns}
    num = df.select_dtypes(include="number").columns.tolist()
    cat = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    kw = {
        "value": ["dealvalue", "revenue", "sales", "amount", "value", "salary", "monthlysalary", "wage", "income", "price", "cost", "total", "profit"],
        "discount": ["discount", "disc", "reduction", "rebate"],
        "booking": ["booking", "advance", "deposit", "payment", "booked"],
        "branch": ["branch", "region", "city", "location", "area", "zone", "territory", "office", "state"],
        "emp": ["employee", "emp", "staff", "salesperson", "agent", "salesemployee", "person", "executive", "seller", "staffid"],
        "status": ["status", "dealstatus", "result", "outcome", "stage", "state"],
        "date": ["date", "month", "year", "time", "period", "created", "closed", "ordered", "start", "end", "joined"],
        "name": ["name", "fullname", "customername", "clientname", "staffname", "empname"],
    }

    def find(keywords, pool):
        return next((c for c in pool if any(k in cl[c] for k in keywords)), None)

    date_candidates = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_candidates.append(col)
        elif any(k in cl[col] for k in kw["date"]):
            date_candidates.append(col)

    return {
        "value_col": find(kw["value"], num),
        "discount_col": find(kw["discount"], num),
        "booking_col": find(kw["booking"], num),
        "branch_col": find(kw["branch"], cat),
        "emp_col": find(kw["emp"], cat),
        "status_col": find(kw["status"], cat),
        "date_col": date_candidates[0] if date_candidates else None,
        "name_col": find(kw["name"], cat),
        "num_cols": num,
        "cat_cols": cat,
    }

def compute_kpis(df, det):
    kpis = {"total_records": int(len(df))}

    vc = det["value_col"]
    sc = det["status_col"]
    dc = det["discount_col"]
    bc = det["booking_col"]
    br = det["branch_col"]
    ec = det["emp_col"]

    num_cols = det.get("num_cols", [])
    cat_cols = det.get("cat_cols", [])

    if vc and vc in df.columns:
        s = df[vc].dropna()
        if not s.empty:
            kpis.update({
                "total_value": round(float(s.sum()), 0),
                "avg_value": round(float(s.mean()), 0),
                "max_value": round(float(s.max()), 0),
                "min_value": round(float(s.min()), 0),
                "value_col": vc
            })
    elif num_cols:
        n1 = next((c for c in num_cols if "id" not in c.lower()), num_cols[0] if num_cols else None)
        if n1:
            s = df[n1].dropna()
            if not s.empty:
                kpis.update({
                    "avg_value": round(float(s.mean()), 2),
                    "max_value": round(float(s.max()), 2),
                    "min_value": round(float(s.min()), 2),
                    "value_col": n1
                })

    if sc and sc in df.columns:
        vc_counts = df[sc].astype(str).value_counts()
        win_vals = [v for v in vc_counts.index if any(k in str(v).lower() for k in ["won", "win", "closed", "success", "complete"])]

        if win_vals and len(df) > 0:
            won = int(vc_counts[win_vals].sum())
            kpis.update({
                "win_rate": round((won / len(df)) * 100, 1),
                "won_count": won
            })

        kpis["status_breakdown"] = {str(k): int(v) for k, v in vc_counts.head(6).items()}
        kpis["status_col"] = sc
    elif cat_cols:
        valid = [c for c in cat_cols if 1 < df[c].nunique() <= 15]
        if valid:
            c1 = valid[0]
            vc_counts = df[c1].astype(str).value_counts()
            if not vc_counts.empty:
                kpis.update({
                    "top_category": str(vc_counts.idxmax()),
                    "top_count": int(vc_counts.max()),
                    "cat_col": c1
                })

    if dc and dc in df.columns:
        s = df[dc].dropna()
        if not s.empty:
            kpis.update({
                "avg_discount": round(float(s.mean()), 0),
                "total_discount": round(float(s.sum()), 0),
                "discount_col": dc
            })

    if bc and bc in df.columns:
        s = df[bc].dropna()
        if not s.empty:
            kpis.update({
                "avg_booking": round(float(s.mean()), 0),
                "booking_col": bc
            })

    if br and br in df.columns:
        kpis["branch_count"] = int(df[br].nunique())
        kpis["branch_col"] = br
        if vc and vc in df.columns and len(df) > 1:
            try:
                grp = df.groupby(br)[vc].sum()
                if not grp.empty:
                    kpis["top_branch"] = str(grp.idxmax())
            except Exception:
                pass
    elif cat_cols:
        valid = [c for c in cat_cols if 10 < df[c].nunique() <= 100]
        if valid:
            c1 = valid[0]
            kpis["distinct_count"] = int(df[c1].nunique())
            kpis["distinct_col"] = c1

    if ec and ec in df.columns:
        kpis["emp_count"] = int(df[ec].nunique())
        kpis["emp_col"] = ec
        if vc and vc in df.columns and len(df) > 1:
            try:
                grp = df.groupby(ec)[vc].sum()
                if not grp.empty:
                    kpis["top_employee"] = str(grp.idxmax())
            except Exception:
                pass

    return kpis

def compute_charts(df, det):
    charts = []

    num_cols = det.get("num_cols", [])
    cat_cols = det.get("cat_cols", [])
    date_col = det.get("date_col")

    valid_cats = [c for c in cat_cols if 1 < df[c].nunique() <= 50]

    # 1. Distribution of top categorical (Doughnut / Pie)
    if valid_cats:
        c1 = valid_cats[0]
        try:
            vc = df[c1].astype(str).value_counts().head(8)
            if not vc.empty:
                charts.append({
                    "id": "chart_cat_1",
                    "type": "doughnut",
                    "title": f"Distribution of {c1}",
                    "labels": list(vc.index),
                    "values": [int(v) for v in vc.values],
                    "icon": "ic-p"
                })
        except Exception:
            pass
            
        # 2. Count by second categorical (Bar)
        if len(valid_cats) > 1:
            c2 = valid_cats[1]
            try:
                vc2 = df[c2].astype(str).value_counts().head(10)
                if not vc2.empty:
                    charts.append({
                        "id": "chart_cat_2",
                        "type": "bar",
                        "title": f"Count by {c2}",
                        "labels": list(vc2.index),
                        "values": [int(v) for v in vc2.values],
                        "icon": "ic-t"
                    })
            except Exception:
                pass

    # 3. Numeric vs Categorical (Bar)
    if valid_cats and num_cols:
        c1 = valid_cats[0]
        n1 = next((n for n in num_cols if "id" not in n.lower()), num_cols[0])
        try:
            grp = df.groupby(c1)[n1].mean().round(2).sort_values(ascending=False).head(10)
            if not grp.empty:
                charts.append({
                    "id": "chart_num_cat",
                    "type": "bar",
                    "title": f"Avg {n1} by {c1}",
                    "labels": list(grp.index.astype(str)),
                    "values": [float(v) for v in grp.values],
                    "icon": "ic-k"
                })
        except Exception:
            pass

    # 4. Time Series
    if date_col and date_col in df.columns:
        try:
            df2 = df.copy()
            df2["_m"] = pd.to_datetime(df2[date_col], errors="coerce").dt.to_period("M").astype(str)
            df2 = df2[df2["_m"] != "NaT"]
            
            g2 = df2.groupby("_m").size().sort_index().tail(12)
            if not g2.empty:
                charts.append({
                    "id": "chart_time_count",
                    "type": "line",
                    "title": "Records over Time",
                    "labels": list(g2.index.astype(str)),
                    "values": [int(v) for v in g2.values],
                    "icon": "ic-a"
                })
                
            if num_cols:
                n1 = next((n for n in num_cols if "id" not in n.lower()), num_cols[0])
                g3 = df2.groupby("_m")[n1].sum().round(2).sort_index().tail(12)
                if not g3.empty:
                    charts.append({
                        "id": "chart_time_num",
                        "type": "line",
                        "title": f"Total {n1} over Time",
                        "labels": list(g3.index.astype(str)),
                        "values": [float(v) for v in g3.values],
                        "icon": "ic-g"
                    })
        except Exception:
            pass
            
    # 5. Scatter if we have 2 numerics and few rows or sample
    if len(num_cols) >= 2:
        n1 = next((n for n in num_cols if "id" not in n.lower()), num_cols[0])
        # Find another numeric that is not id, default to the second element
        n2_candidates = (n for n in num_cols if n != n1 and "id" not in n.lower())
        n2 = next(n2_candidates, next((n for n in num_cols if n != n1), None))
        
        if n2:
            try:
                s_df = df[[n1, n2]].dropna().head(300)
                if not s_df.empty:
                    charts.append({
                        "id": "chart_scatter",
                        "type": "scatter",
                        "title": f"{n1} vs {n2}",
                        "labels": [],
                        "values": [{"x": float(r[n1]), "y": float(r[n2])} for _, r in s_df.iterrows()],
                        "icon": "ic-r",
                        "x_label": n1,
                        "y_label": n2
                    })
            except Exception:
                pass

    return charts

def auto_insights(df, det, kpis):
    ins = []

    vc = det["value_col"]
    br = det["branch_col"]
    ec = det["emp_col"]
    dc = det["discount_col"]
    dtc = det["date_col"]

    if vc and vc in df.columns:
        s = df[vc].dropna()
        if not s.empty:
            ins.append(f"Total {vc}: {s.sum():,.0f}")
            ins.append(f"Average {vc}: {s.mean():,.0f}")
            ins.append(f"Highest {vc}: {s.max():,.0f}")

    if br and vc and br in df.columns and vc in df.columns and len(df) > 1:
        try:
            top = df.groupby(br)[vc].sum()
            if not top.empty:
                ins.append(f"Top {br}: {top.idxmax()} ({top.max():,.0f})")
        except Exception:
            pass

    if ec and vc and ec in df.columns and vc in df.columns and len(df) > 1:
        try:
            top = df.groupby(ec)[vc].sum()
            if not top.empty:
                ins.append(f"Top {ec}: {top.idxmax()} ({top.max():,.0f})")
        except Exception:
            pass

    if "win_rate" in kpis:
        ins.append(f"Win Rate: {kpis['win_rate']}% ({kpis['won_count']} won / {len(df)} total)")

    if dc and dc in df.columns:
        s = df[dc].dropna()
        if not s.empty:
            ins.append(f"Avg {dc}: {s.mean():,.0f}")

    if dtc and dtc in df.columns:
        try:
            df2 = df.copy()
            df2["_m"] = pd.to_datetime(df2[dtc], errors="coerce").dt.to_period("M").astype(str)
            df2 = df2[df2["_m"] != "NaT"]
            if not df2.empty:
                best = df2.groupby("_m").size().idxmax()
                ins.append(f"Busiest Month: {best}")
        except Exception:
            pass

    miss = int(df.isnull().sum().sum())
    ins.append(f"Missing Values: {miss}" if miss else "Data Quality: Clean")

    return ins

def build_col_stats(df, num_cols):
    col_stats = {}
    for c in num_cols:
        s = df[c].dropna()
        if len(s) == 0:
            col_stats[c] = {
                "mean": None,
                "min": None,
                "max": None,
                "median": None,
                "missing": int(df[c].isnull().sum()),
                "total": None
            }
        else:
            col_stats[c] = {
                "mean": round(float(s.mean()), 2),
                "min": round(float(s.min()), 2),
                "max": round(float(s.max()), 2),
                "median": round(float(s.median()), 2),
                "missing": int(df[c].isnull().sum()),
                "total": round(float(s.sum()), 0)
            }
    return col_stats

def build_filter_options(df, det):
    filter_options = {}
    for col in det["cat_cols"]:
        try:
            unique_count = int(df[col].nunique(dropna=True))
            if 1 < unique_count <= 60:
                values = sorted(df[col].dropna().astype(str).unique().tolist())
                filter_options[col] = values
        except Exception:
            pass
    return filter_options
