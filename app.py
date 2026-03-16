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
    csv = """staffid,staffname,businessunit,department,startdate,enddate,gender,monthlysalary,region,city,state
E0001,Aditya Ghosh,Support,Sales,2021-01-24,NaT,Female,40000,West,Surat,Gujarat
E0002,Karan Nair,Marketing,Ops,2019-01-09,NaT,Female,67000,North,Jaipur,Rajasthan
E0003,Kiara Ghosh,HR,Customer Success,2023-11-02,NaT,Male,73500,North,Jaipur,Rajasthan
E0004,Isha Sharma,Support,Growth,2019-05-30,NaT,Female,71500,South,Mysuru,Karnataka
E0005,Aadhya Malhotra,Finance,IT,2022-06-04,NaT,Female,96000,North,Lucknow,Uttar Pradesh
E0006,Sneha Malhotra,Finance,People,2025-08-04,2026-07-06,Female,79500,East,Kolkata,West Bengal
E0007,Aditya Malhotra,Finance,Accounts,2025-06-27,NaT,Male,113000,North,New Delhi,Delhi
E0008,Rahul Bose,Engineering,Product,2020-11-19,2024-08-28,Male,135000,East,Kolkata,West Bengal
E0009,Muhammad Mehta,Support,IT,2024-03-29,NaT,Male,62000,South,Kochi,Kerala
E0010,Aditya Kapoor,Sales,Product,2022-11-25,NaT,Female,77000,North,Lucknow,Uttar Pradesh
E0011,Sara Patel,Engineering,Customer Success,2021-11-01,NaT,Male,131000,North,Noida,Uttar Pradesh
E0012,Meera Malhotra,Engineering,Ops,2019-02-13,NaT,Male,124000,South,Hyderabad,Telangana
E0013,Neha Das,Engineering,Growth,2021-01-04,2021-04-22,Male,91000,North,Jaipur,Rajasthan
E0014,Neha Mehta,Sales,Growth,2023-03-20,NaT,Female,88500,North,New Delhi,Delhi
E0015,Priya Chatterjee,Sales,Customer Success,2025-11-21,2027-02-11,Male,95000,West,Ahmedabad,Gujarat
E0016,Karan Das,Operations,Sales,2018-01-15,NaT,Female,62000,West,Surat,Gujarat
E0017,Sara Sharma,Support,Customer Success,2023-09-28,NaT,Female,69000,South,Chennai,Tamil Nadu
E0018,Vihaan Iyer,Sales,Sales,2024-10-08,NaT,Female,61000,South,Chennai,Tamil Nadu
E0019,Sneha Gupta,HR,People,2025-08-10,NaT,Male,101500,West,Nagpur,Maharashtra
E0020,Isha Iyer,Finance,Accounts,2021-05-22,2024-12-19,Female,120000,South,Mysuru,Karnataka
E0021,Sara Reddy,Support,IT,2018-12-13,NaT,Female,60000,South,Hyderabad,Telangana
E0022,Diya Bose,Operations,Product,2020-03-02,2022-05-15,Female,87500,South,Hyderabad,Telangana
E0023,Reyansh Bose,Marketing,Product,2024-10-20,NaT,Male,37000,South,Mysuru,Karnataka
E0024,Pooja Mehta,Operations,Growth,2025-01-08,NaT,Male,57000,South,Hyderabad,Telangana
E0025,Karan Kulkarni,HR,IT,2023-02-04,2027-12-22,Male,69000,West,Surat,Gujarat
E0026,Kiara Das,Engineering,IT,2018-12-25,2023-09-19,Male,98500,South,Coimbatore,Tamil Nadu
E0027,Aadhya Verma,Support,Customer Success,2018-12-23,2021-10-29,Male,79500,North,New Delhi,Delhi
E0028,Rohit Malhotra,Marketing,Product,2021-10-28,NaT,Male,125000,North,New Delhi,Delhi
E0029,Aarohi Das,Sales,Customer Success,2022-09-15,NaT,Male,66500,East,Kolkata,West Bengal
E0030,Isha Jain,Marketing,Customer Success,2018-04-05,NaT,Female,68500,North,New Delhi,Delhi
E0031,Isha Das,Sales,Accounts,2023-06-28,NaT,Male,94500,North,Jaipur,Rajasthan
E0032,Muhammad Das,Marketing,Product,2022-05-10,NaT,Female,117000,North,New Delhi,Delhi
E0033,Aditya Verma,Marketing,Product,2018-02-05,2019-01-03,Female,60500,South,Hyderabad,Telangana
E0034,Aarohi Das,Sales,IT,2018-10-07,NaT,Male,95000,North,New Delhi,Delhi
E0035,Vikram Malhotra,Finance,Accounts,2021-08-20,NaT,Female,131500,South,Mysuru,Karnataka
E0036,Vivaan Singh,Sales,Product,2023-10-12,NaT,Male,57000,West,Surat,Gujarat
E0037,Sneha Jain,Finance,People,2024-08-07,NaT,Female,69500,South,Chennai,Tamil Nadu
E0038,Rohit Sharma,Support,Sales,2021-06-13,NaT,Male,56500,South,Chennai,Tamil Nadu
E0039,Arnav Singh,Sales,Accounts,2020-05-12,NaT,Female,79500,North,Jaipur,Rajasthan
E0040,Saanvi Reddy,HR,Sales,2025-09-29,2029-11-26,Male,91500,South,Thiruvananthapuram,Kerala
E0041,Arjun Rao,Support,Ops,2021-09-01,NaT,Male,70000,West,Ahmedabad,Gujarat
E0042,Neha Gupta,HR,Sales,2020-07-10,NaT,Male,39500,South,Thiruvananthapuram,Kerala
E0043,Priya Patel,Sales,Product,2021-07-15,NaT,Female,100500,North,Noida,Uttar Pradesh
E0044,Ananya Reddy,Finance,Product,2018-02-05,2020-08-03,Female,83500,South,Hyderabad,Telangana
E0045,Aditya Rao,Finance,Growth,2020-05-31,2022-11-10,Male,120500,North,Jaipur,Rajasthan
E0046,Meera Kulkarni,Operations,Sales,2022-02-14,NaT,Male,67500,North,New Delhi,Delhi
E0047,Isha Sharma,HR,People,2025-05-06,NaT,Male,62500,South,Hyderabad,Telangana
E0048,Vikram Gupta,Support,Accounts,2023-11-27,NaT,Male,76000,East,Kolkata,West Bengal
E0049,Rahul Ghosh,Finance,Accounts,2024-04-01,NaT,Male,110500,South,Kochi,Kerala
E0050,Karan Verma,HR,Ops,2022-05-11,NaT,Male,84000,North,Noida,Uttar Pradesh
E0051,Neha Kulkarni,Support,Growth,2018-11-20,NaT,Female,31500,South,Hyderabad,Telangana
E0052,Reyansh Patel,Support,People,2021-02-15,NaT,Female,76000,West,Ahmedabad,Gujarat
E0053,Ananya Joshi,Finance,Growth,2025-12-22,NaT,Male,74500,South,Coimbatore,Tamil Nadu
E0054,Myra Patel,Sales,IT,2018-05-10,NaT,Female,94500,North,Lucknow,Uttar Pradesh
E0055,Myra Reddy,Operations,Customer Success,2025-01-12,NaT,Male,51500,East,Kolkata,West Bengal
E0056,Vivaan Bose,Finance,Accounts,2023-06-23,2024-08-15,Female,104000,East,Kolkata,West Bengal
E0057,Aarohi Iyer,Engineering,Ops,2020-04-09,NaT,Male,104000,North,Jaipur,Rajasthan
E0058,Arjun Das,Operations,Customer Success,2024-03-22,NaT,Female,54500,West,Ahmedabad,Gujarat
E0059,Neha Chatterjee,Marketing,Growth,2024-09-10,NaT,Female,99500,West,Pune,Maharashtra
E0060,Rahul Malhotra,Sales,Growth,2018-05-07,NaT,Female,95000,South,Coimbatore,Tamil Nadu
E0061,Pooja Rao,Marketing,Accounts,2018-11-07,NaT,Male,72500,South,Bengaluru,Karnataka
E0062,Neha Bose,Sales,Sales,2023-04-29,NaT,Female,119000,North,Jaipur,Rajasthan
E0063,Ananya Reddy,Marketing,People,2025-12-28,NaT,Male,76000,West,Nagpur,Maharashtra
E0064,Meera Reddy,HR,Product,2018-03-12,NaT,Male,75500,South,Coimbatore,Tamil Nadu
E0065,Aadhya Reddy,Sales,Customer Success,2023-04-29,NaT,Female,95500,East,Kolkata,West Bengal
E0066,Ayaan Ghosh,Support,Product,2019-02-03,NaT,Male,52500,West,Ahmedabad,Gujarat
E0067,Pooja Singh,Engineering,Product,2023-03-15,NaT,Female,129500,East,Kolkata,West Bengal
E0068,Diya Nair,Operations,Product,2022-12-26,NaT,Female,69500,North,Noida,Uttar Pradesh
E0069,Aditya Singh,Engineering,Ops,2018-09-22,NaT,Male,112000,South,Thiruvananthapuram,Kerala
E0070,Myra Sharma,Support,Accounts,2023-06-12,NaT,Female,92500,South,Bengaluru,Karnataka
E0071,Karan Iyer,Marketing,Customer Success,2025-03-05,NaT,Male,64000,South,Thiruvananthapuram,Kerala
E0072,Reyansh Verma,Marketing,People,2024-11-27,NaT,Male,36500,West,Surat,Gujarat
E0073,Isha Malhotra,Finance,IT,2018-03-26,NaT,Male,127000,West,Surat,Gujarat
E0074,Vihaan Kapoor,Finance,Sales,2019-12-20,NaT,Female,146000,West,Nagpur,Maharashtra
E0075,Rahul Bose,Operations,Sales,2019-12-05,NaT,Male,72000,West,Surat,Gujarat
E0076,Myra Mehta,Marketing,IT,2021-08-17,NaT,Male,46500,North,Jaipur,Rajasthan
E0077,Ayaan Kapoor,Marketing,Customer Success,2019-02-22,NaT,Male,79000,East,Kolkata,West Bengal
E0078,Muhammad Nair,Sales,Accounts,2024-10-31,NaT,Female,84500,West,Mumbai,Maharashtra
E0079,Aarohi Joshi,Operations,Customer Success,2024-03-18,NaT,Female,76000,East,Kolkata,West Bengal
E0080,Rahul Chatterjee,Sales,Sales,2024-06-19,NaT,Female,67500,South,Thiruvananthapuram,Kerala
E0081,Nikhil Joshi,HR,Sales,2019-08-02,2021-07-19,Female,86500,North,Jaipur,Rajasthan
E0082,Isha Nair,Marketing,Growth,2018-02-12,2021-11-05,Female,95500,South,Bengaluru,Karnataka
E0083,Arjun Mehta,Engineering,Sales,2025-04-13,NaT,Male,90000,South,Thiruvananthapuram,Kerala
E0084,Rohit Bose,Marketing,Sales,2020-04-23,2024-09-06,Male,99000,South,Kochi,Kerala
E0085,Saanvi Mehta,Finance,People,2023-04-01,NaT,Female,136000,South,Bengaluru,Karnataka
E0086,Priya Reddy,Engineering,Customer Success,2025-04-10,NaT,Female,123500,North,Lucknow,Uttar Pradesh
E0087,Aditya Nair,HR,Ops,2018-04-16,2022-06-28,Male,68500,South,Hyderabad,Telangana
E0088,Ayaan Khan,Finance,Sales,2018-01-16,NaT,Male,92000,East,Kolkata,West Bengal
E0089,Karan Gupta,Marketing,Growth,2018-01-16,NaT,Female,97500,South,Hyderabad,Telangana
E0090,Rohit Singh,Operations,Growth,2019-04-02,2021-12-26,Male,45500,North,Jaipur,Rajasthan
E0091,Isha Bose,Support,Growth,2018-05-31,NaT,Female,68500,South,Thiruvananthapuram,Kerala
E0092,Rohit Jain,Support,Customer Success,2024-07-23,NaT,Male,57000,North,Jaipur,Rajasthan
E0093,Sneha Jain,Marketing,Product,2022-11-05,NaT,Male,122000,South,Kochi,Kerala
E0094,Diya Khan,Operations,Customer Success,2025-11-02,2027-10-25,Female,76500,South,Kochi,Kerala
E0095,Vikram Rao,HR,IT,2024-05-30,NaT,Male,108500,West,Nagpur,Maharashtra
E0096,Arnav Verma,HR,Growth,2025-09-14,2027-12-20,Male,53000,South,Coimbatore,Tamil Nadu
E0097,Sai Singh,Sales,Accounts,2019-11-28,NaT,Female,101500,North,Jaipur,Rajasthan
E0098,Rahul Iyer,HR,Growth,2020-04-20,NaT,Female,81500,East,Kolkata,West Bengal
E0099,Kiara Singh,Support,Ops,2024-10-17,NaT,Male,47500,West,Pune,Maharashtra
E0100,Arjun Rao,HR,Ops,2019-12-17,NaT,Male,78500,West,Nagpur,Maharashtra"""

    return send_file(
        io.BytesIO(csv.encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name="sample_sales.csv"
    )

if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)
