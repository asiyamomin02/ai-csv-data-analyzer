from flask import Flask, render_template, request, jsonify, send_file
from flask.json.provider import DefaultJSONProvider
import pandas as pd
import io
import numpy as np
import os
import sqlite3
from dotenv import load_dotenv

from services.data_processing import (
    read_uploaded_file,
    try_parse_dates,
    smart_detect,
    compute_kpis,
    compute_charts,
    auto_insights,
    build_col_stats,
    build_filter_options
)
from services.ai_service import call_ai, get_api_key

load_dotenv()

class SafeJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        try:
            if pd.isna(obj):
                return None
        except Exception:
            pass
        return super().default(obj)


app = Flask(__name__)
app.json = SafeJSONProvider(app)

df_store = {}
last_filtered = {}
last_analysis = {}
db_conn = sqlite3.connect(':memory:', check_same_thread=False)

def make_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    try:
        df = read_uploaded_file(file)
        df = try_parse_dates(df)

        if df.empty:
            return jsonify({"error": "Uploaded file is empty"}), 400

        df_store["data"] = df
        last_filtered["data"] = df.copy()
        last_analysis["filename"] = file.filename

        try:
            df.to_sql("data_table", db_conn, index=False, if_exists="replace")
        except Exception as e:
            print("SQLite save error:", e)

        det = smart_detect(df)
        kpis = compute_kpis(df, det)
        charts = compute_charts(df, det)
        insights = auto_insights(df, det, kpis)
        num_cols = det["num_cols"]
        col_stats = build_col_stats(df, num_cols)

        try:
            sample_df = df.sample(min(len(df), 10000)) if len(df) > 10000 else df
            describe_txt = sample_df.describe(include="all").fillna("").astype(str).head(10).to_string()
        except Exception:
            describe_txt = "Summary unavailable"

        analysis = call_ai(
            "You are a data analyst. Give a short plain-text summary in max 5 lines. No markdown.",
            f"Rows: {df.shape[0]}\nColumns: {list(df.columns)}\nSummary:\n{describe_txt}\nGive key findings and one recommendation.",
            260
        )
        last_analysis["text"] = analysis

        missing = {c: int(v) for c, v in df.isnull().sum().items() if int(v) > 0}
        filter_options = build_filter_options(df, det)

        date_range = {}
        if det["date_col"] and det["date_col"] in df.columns:
            try:
                dates = pd.to_datetime(df[det["date_col"]], errors="coerce").dropna()
                if not dates.empty:
                    date_range = {
                        "min": str(dates.min().date()),
                        "max": str(dates.max().date()),
                        "col": det["date_col"]
                    }
            except Exception:
                pass

        chat_suggestions = ["Any missing data?"]
        if num_cols:
            n1 = next((c for c in num_cols if "id" not in c.lower()), num_cols[0])
            chat_suggestions.insert(0, f"Top 5 by {n1}")
            chat_suggestions.append(f"Average {n1}?")
            if len(num_cols) > 1:
                n2 = next((c for c in num_cols if c != n1 and "id" not in c.lower()), num_cols[1])
                chat_suggestions.append(f"Highest {n2}?")
        if det["cat_cols"]:
            chat_suggestions.append(f"Best {det['cat_cols'][0]}?")
            if len(det["cat_cols"]) > 1:
                chat_suggestions.append(f"List {det['cat_cols'][1]}")
                
        response_data = {
            "success": True,
            "filename": file.filename,
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "total_rows": int(df.shape[0]),
            "filtered_rows": int(df.shape[0]),
            "col_names": list(df.columns),
            "analysis": analysis,
            "col_stats": col_stats,
            "charts": charts,
            "kpis": kpis,
            "insights": insights,
            "missing": missing,
            "detected": det,
            "filter_options": filter_options,
            "date_range": date_range,
            "chat_suggestions": chat_suggestions[:6],
            "table_data": df.head(300).fillna("").to_dict(orient="records"),
            "table_cols": list(df.columns),
            "all_numeric": num_cols,
            "all_cat": det["cat_cols"],
            "ai_enabled": bool(get_api_key()),
        }

        return jsonify(make_json_safe(response_data))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/filter", methods=["POST"])
def apply_filter():
    if "data" not in df_store:
        return jsonify({"error": "Upload file first"}), 400

    body = request.json or {}
    base_df = df_store["data"]
    df = base_df.copy()

    for col, val in body.get("filters", {}).items():
        if col in df.columns and val not in [None, "", "__ALL__"]:
            df = df[df[col].astype(str) == str(val)]

    dr = body.get("date_range", {})
    date_col = dr.get("col")
    if date_col and date_col in df.columns:
        try:
            date_series = pd.to_datetime(df[date_col], errors="coerce")
            if dr.get("from"):
                df = df[date_series >= pd.to_datetime(dr["from"])]
                date_series = pd.to_datetime(df[date_col], errors="coerce")
            if dr.get("to"):
                df = df[date_series <= pd.to_datetime(dr["to"])]
        except Exception:
            pass

    last_filtered["data"] = df.copy()

    det = smart_detect(df)
    kpis = compute_kpis(df, det)
    charts = compute_charts(df, det)
    insights = auto_insights(df, det, kpis)
    col_stats = build_col_stats(df, det["num_cols"])

    response_data = {
        "rows": int(len(df)),
        "total_rows": int(len(base_df)),
        "filtered_rows": int(len(df)),
        "kpis": kpis,
        "charts": charts,
        "insights": insights,
        "col_stats": col_stats,
        "table_cols": list(df.columns),
        "table_data": df.head(300).fillna("").to_dict(orient="records")
    }

    return jsonify(make_json_safe(response_data))

@app.route("/chart-data", methods=["POST"])
def chart_data_api():
    if "data" not in df_store:
        return jsonify({"error": "Upload file first"}), 400

    body = request.json or {}
    df = last_filtered.get("data", df_store["data"])

    cat_col = body.get("cat")
    num_col = body.get("num")
    agg = body.get("agg", "mean")

    if not cat_col or cat_col not in df.columns:
        return jsonify({"error": "Invalid category column"}), 400

    if agg != "count" and (not num_col or num_col not in df.columns):
        return jsonify({"error": "Invalid numeric column"}), 400

    try:
        fn = {"mean": "mean", "sum": "sum", "count": "count"}.get(agg, "mean")

        if agg == "count":
            grp = df.groupby(cat_col).size().sort_values(ascending=False).head(10)
        else:
            grp = getattr(df.groupby(cat_col)[num_col], fn)().round(0).sort_values(ascending=False).head(10)

        return jsonify(make_json_safe({
            "labels": list(grp.index.astype(str)),
            "values": list(grp.values)
        }))
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Question is required"}), 400

    if "data" not in df_store:
        return jsonify({"error": "Upload file first"}), 400

    df = last_filtered.get("data", df_store["data"])
    api_key_set = bool(get_api_key())
    
    if not api_key_set:
        return jsonify({"error": "AI is disabled due to missing API key."}), 400

    try:
        schema_query = "PRAGMA table_info(data_table);"
        schema_df = pd.read_sql(schema_query, db_conn)
        schema_str = ", ".join([f"'{r['name']}' ({r['type']})" for _, r in schema_df.iterrows()])
        
        # Get a small sample of data to help AI understand the actual values format
        sample_df = df.head(3)
        sample_str = sample_df.to_string(index=False)
        
        sql_sys = (
            "You are a SQL expert. Given the table schema and a sample of the data, write a valid SQLite query to answer the user's question. "
            "Output your SQL wrapped in ```sql and ``` blocks. "
            "Wrap column names in double quotes. "
            "If asked about a specific person or thing, use a WHERE clause with LIKE or exact match, do NOT just SUM the entire table blindly. "
            "Table name is 'data_table'."
        )
        sql_query = call_ai(sql_sys, f"Schema:\ndata_table({schema_str})\n\nData Sample:\n{sample_str}\n\nQuestion: {question}", 200).strip()
        
        # Robust extraction of the SQL query from potential conversational filler
        import re
        sql_match = re.search(r"```(?:sql|sqlite)?\n?(.*?)\n?```", sql_query, re.IGNORECASE | re.DOTALL)
        if sql_match:
            sql_query = sql_match.group(1).strip()
        else:
            # Fallback if no code blocks were used, try to strip common prefixes
            if sql_query.lower().startswith("here is"):
                lines = sql_query.split("\n")
                sql_query = "\n".join([line for line in lines if not line.lower().startswith("here is") and not line.lower().startswith("i have")])
            sql_query = sql_query.strip()
            
        print(f"Extracted SQL: {sql_query}")
        
        try:
            result_df = pd.read_sql(sql_query, db_conn)
            result_str = result_df.to_string(index=False)
            if len(result_str) > 2000:
                result_str = result_str[:2000] + "\n... (truncated)"
        except Exception as e:
            result_str = f"Error executing query: {e}"
            print(result_str)
            
        nl_sys = (
            "You are a helpful data analyst. Answer the user's question concisely using the provided SQL execution result. "
            "Plain text only. No markdown. If there was an error, output exactly: 'Error: ' followed by the error message, and the SQL query you tried."
        )
        final_answer = call_ai(nl_sys, f"Question: {question}\n\nSQL Query:\n{sql_query}\n\nSQL Result:\n{result_str}", 400).strip()
        
        has_filter = False
        filter_count = int(len(df))
        
        fq = call_ai(
            "Return ONLY a valid pandas DataFrame `.query()` string or the word 'none'. You MUST wrap column names containing spaces in backticks (e.g., `Sales Employee`). No markdown.",
            f"Columns:{list(df.columns)}, dtypes:{dict(df.dtypes.astype(str))}\nQ:{question}",
            70
        ).strip().strip("\"'").strip("`")

        if fq.lower() not in ["none", "", "sql"] and len(fq) < 300:
            try:
                filtered = df.query(fq)
                last_filtered["data"] = filtered
                has_filter = True
                filter_count = int(len(filtered))
            except Exception:
                pass
                
        return jsonify(make_json_safe({
            "answer": final_answer,
            "has_filter": has_filter,
            "filter_count": int(filter_count),
            "ai_enabled": api_key_set
        }))
        
    except Exception as e:
        print(f"Error in ask API: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/download-filtered")
def download_filtered():
    df = last_filtered.get("data", df_store.get("data"))
    if df is None:
        return "No data", 400

    out = io.StringIO()
    df.to_csv(out, index=False)
    out.seek(0)

    return send_file(
        io.BytesIO(out.getvalue().encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name="filtered_data.csv"
    )

@app.route("/download-report")
def download_report():
    df = last_filtered.get("data", df_store.get("data"))
    if df is None:
        return "No data", 400

    det = smart_detect(df)
    kpis = compute_kpis(df, det)
    ins = auto_insights(df, det, kpis)

    try:
        stats_txt = df.describe(include="all").fillna("").astype(str).to_string()
    except Exception:
        stats_txt = "Statistics unavailable"

    report = f"""DATALENS — AI ANALYTICS REPORT
{'=' * 60}
File: {last_analysis.get('filename', 'data.csv')}
Rows: {df.shape[0]} | Cols: {df.shape[1]}
Columns: {', '.join(df.columns)}
{'=' * 60}
AI ANALYSIS
{last_analysis.get('text', '')}
{'=' * 60}
KEY INSIGHTS
{chr(10).join('• ' + i for i in ins)}
{'=' * 60}
STATISTICS
{stats_txt}
{'=' * 60}
SAMPLE (first 20 rows)
{df.head(20).to_string(index=False)}
{'=' * 60}
Generated by DataLens AI"""

    return send_file(
        io.BytesIO(report.encode()),
        mimetype="text/plain",
        as_attachment=True,
        download_name="datalens_report.txt"
    )

@app.route("/sample")
def sample():
    csv = """deal_id,deal_date,branch,sales_employee_id,deal_status,deal_value,discount_value,booking_amount
D0000001,2025-01-15,Mumbai,E0043,Won,1500000,75000,150000
D0000002,2025-01-22,Delhi,E0042,Won,2200000,154000,220000
D0000003,2025-02-10,Pune,E0051,Lost,900000,63000,90000
D0000004,2025-02-18,Mumbai,E0060,Won,1850000,129500,185000
D0000005,2025-03-05,Hyderabad,E0011,Won,1200000,84000,120000
D0000006,2025-03-12,Delhi,E0054,Won,750000,52500,75000
D0000007,2025-03-25,Bangalore,E0053,Lost,980000,68600,98000
D0000008,2025-04-08,Mumbai,E0043,Won,1650000,115500,165000
D0000009,2025-04-15,Pune,E0042,Won,1100000,77000,110000
D0000010,2025-05-02,Hyderabad,E0060,Lost,850000,59500,85000
D0000011,2025-05-18,Delhi,E0011,Won,1950000,136500,195000
D0000012,2025-06-01,Mumbai,E0051,Won,2100000,147000,210000
D0000013,2025-06-14,Bangalore,E0054,Won,1300000,91000,130000
D0000014,2025-07-03,Pune,E0053,Lost,760000,53200,76000
D0000015,2025-07-20,Delhi,E0043,Won,1750000,122500,175000
D0000016,2025-08-05,Hyderabad,E0042,Won,1400000,98000,140000
D0000017,2025-08-19,Mumbai,E0060,Won,1900000,133000,190000
D0000018,2025-09-03,Bangalore,E0011,Lost,670000,46900,67000
D0000019,2025-09-22,Delhi,E0051,Won,2050000,143500,205000
D0000020,2025-10-10,Pune,E0054,Won,1150000,80500,115000"""

    return send_file(
        io.BytesIO(csv.encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name="sample_sales.csv"
    )

if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)