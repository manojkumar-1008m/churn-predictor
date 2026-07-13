import csv
import json
from datetime import datetime
from io import BytesIO, StringIO

import pandas as pd

from django.contrib.auth import login
from django.contrib.auth.models import User
from django.db import IntegrityError, transaction
from django.http import HttpResponse
from django.shortcuts import redirect, render

from .forms import ChurnPredictionForm, DatasetUploadForm, RegisterForm
from .services import (
    get_user_history_entry_by_id,
    get_user_analysis_history_export_rows,
    get_user_analysis_history,
    log_prediction_event,
    predict_churn_customers_from_dataset,
    predict_churn,
    store_registered_user_details,
    train_from_uploaded_dataset,
)


def _build_database_scan_tabular_rows(scan_result: dict) -> list[dict]:
    churn_customers = list(scan_result.get("churn_customers", []))
    churn_details = list(scan_result.get("churn_details", []))
    identifier_column = str(scan_result.get("identifier_column", "identifier"))
    churn_count = int(scan_result.get("churn_count", 0))

    if churn_details:
        rows = []
        for detail in churn_details:
            rows.append(
                {
                    "identifier_column": identifier_column,
                    "churn_count": churn_count,
                    "churn_customer": str(detail.get("customer", "")),
                    "risk": str(detail.get("risk", "")),
                    "probability_percent": float(detail.get("probability_percent", 0)),
                }
            )
        return rows

    if churn_customers:
        return [
            {
                "identifier_column": identifier_column,
                "churn_count": churn_count,
                "churn_customer": str(customer),
                "risk": "",
                "probability_percent": 0,
            }
            for customer in churn_customers
        ]

    return [
        {
            "identifier_column": identifier_column,
            "churn_count": churn_count,
            "churn_customer": "",
            "risk": "",
            "probability_percent": 0,
        }
    ]


def register(request):
    if request.user.is_authenticated:
        return redirect("dashboard")

    if request.method == "POST":
        form = RegisterForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data.get("username", "")
            phone_number = form.cleaned_data.get("phone_number", "")
            if User.objects.filter(username__iexact=username).exists():
                form.add_error("username", "A user with that username already exists.")
                return render(request, "registration/register.html", {"form": form})

            try:
                with transaction.atomic():
                    user = form.save()
                    store_registered_user_details(user, phone_number=phone_number)
            except IntegrityError:
                form.add_error(
                    "username",
                    "That username is already taken. Please choose another.",
                )
                return render(request, "registration/register.html", {"form": form})
            except Exception:
                form.add_error(
                    None,
                    "Could not save user details to MongoDB. Please ensure MongoDB is running and try again.",
                )
                return render(request, "registration/register.html", {"form": form})
            login(request, user)
            return redirect("dashboard")
    else:
        form = RegisterForm()

    return render(request, "registration/register.html", {"form": form})


def home(request):
    if request.user.is_authenticated:
        return redirect("dashboard")
    return render(request, "predictor/landing.html")


def dashboard(request):
    if not request.user.is_authenticated:
        return redirect("home")
    return render(request, "predictor/dashboard.html")


def analysis_history(request):
    if not request.user.is_authenticated:
        return redirect("home")

    history = {"total_count": 0, "items": []}
    error_message = None
    try:
        history = get_user_analysis_history(request.user, limit=None)
    except Exception:
        error_message = "Could not load history right now. Please try again shortly."

    return render(
        request,
        "predictor/history.html",
        {
            "history": history,
            "error_message": error_message,
        },
    )


def analysis_history_download(request):
    if not request.user.is_authenticated:
        return redirect("home")

    export_format = str(request.GET.get("format", "csv")).strip().lower()
    if export_format not in {"csv", "json", "txt", "xlsx"}:
        export_format = "csv"

    try:
        rows = get_user_analysis_history_export_rows(request.user)
    except Exception:
        return HttpResponse(
            "Could not export report right now. Please try again shortly.",
            status=503,
            content_type="text/plain",
        )

    timestamp_tag = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_username = "".join(ch for ch in str(request.user.username) if ch.isalnum() or ch in {"_", "-"}) or "user"
    base_filename = f"history_report_{safe_username}_{timestamp_tag}"

    if export_format == "json":
        payload = {
            "generated_at_utc": datetime.utcnow().isoformat() + "Z",
            "username": str(request.user.username),
            "total_records": len(rows),
            "history": rows,
        }
        response = HttpResponse(json.dumps(payload, indent=2), content_type="application/json")
        response["Content-Disposition"] = f'attachment; filename="{base_filename}.json"'
        return response

    if export_format == "txt":
        lines = [
            f"History Report for {request.user.username}",
            f"Generated (UTC): {datetime.utcnow().isoformat()}Z",
            f"Total Records: {len(rows)}",
            "",
        ]
        for index, row in enumerate(rows, start=1):
            lines.append(f"Record {index}")
            lines.append(f"Timestamp: {row.get('timestamp_utc', '')}")
            lines.append(f"Analysis Type: {row.get('analysis_type', '')}")
            lines.append(f"Analyzed Data: {json.dumps(row.get('analyzed_data', {}), ensure_ascii=True)}")
            lines.append(f"Result Data: {json.dumps(row.get('result_data', {}), ensure_ascii=True)}")
            lines.append("-" * 40)
        response = HttpResponse("\n".join(lines), content_type="text/plain")
        response["Content-Disposition"] = f'attachment; filename="{base_filename}.txt"'
        return response

    if export_format == "xlsx":
        excel_rows = []
        for row in rows:
            excel_rows.append(
                {
                    "timestamp_utc": row.get("timestamp_utc", ""),
                    "analysis_type": row.get("analysis_type", ""),
                    "analyzed_data": json.dumps(row.get("analyzed_data", {}), ensure_ascii=True),
                    "result_data": json.dumps(row.get("result_data", {}), ensure_ascii=True),
                }
            )

        output = BytesIO()
        pd.DataFrame(excel_rows).to_excel(output, index=False, sheet_name="History")
        response = HttpResponse(
            output.getvalue(),
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        response["Content-Disposition"] = f'attachment; filename="{base_filename}.xlsx"'
        return response

    csv_buffer = StringIO()
    fieldnames = ["timestamp_utc", "analysis_type", "analyzed_data", "result_data"]
    writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "timestamp_utc": row.get("timestamp_utc", ""),
                "analysis_type": row.get("analysis_type", ""),
                "analyzed_data": json.dumps(row.get("analyzed_data", {}), ensure_ascii=True),
                "result_data": json.dumps(row.get("result_data", {}), ensure_ascii=True),
            }
        )

    response = HttpResponse(csv_buffer.getvalue(), content_type="text/csv")
    response["Content-Disposition"] = f'attachment; filename="{base_filename}.csv"'
    return response


def analysis_history_entry_download(request):
    if not request.user.is_authenticated:
        return redirect("home")

    entry_id = str(request.GET.get("entry_id", "")).strip()
    if not entry_id:
        return HttpResponse("Missing history entry id.", status=400, content_type="text/plain")

    entry = get_user_history_entry_by_id(request.user, entry_id)
    if not entry:
        return HttpResponse("History entry not found.", status=404, content_type="text/plain")

    export_format = str(request.GET.get("format", "csv")).strip().lower()
    if export_format not in {"csv", "json", "txt", "xlsx"}:
        export_format = "csv"

    timestamp_tag = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_username = "".join(ch for ch in str(request.user.username) if ch.isalnum() or ch in {"_", "-"}) or "user"
    base_filename = f"history_entry_{safe_username}_{timestamp_tag}"

    analysis_type = str(entry.get("analysis_type", "analysis"))
    result_data = entry.get("result_data", {}) or {}

    if analysis_type == "database_scan":
        scan_result = {
            "identifier_column": str(result_data.get("identifier_column", "identifier")),
            "churn_count": int(result_data.get("churn_count", 0)),
            "churn_customers": list(result_data.get("churn_customers", [])),
        }

        if export_format == "json":
            payload = {
                "generated_at_utc": datetime.utcnow().isoformat() + "Z",
                "username": str(request.user.username),
                "report": scan_result,
            }
            response = HttpResponse(json.dumps(payload, indent=2), content_type="application/json")
            response["Content-Disposition"] = f'attachment; filename="{base_filename}.json"'
            return response

        if export_format == "txt":
            lines = [
                f"Database Scan Report for {request.user.username}",
                f"Generated (UTC): {datetime.utcnow().isoformat()}Z",
                f"Identifier Column: {scan_result.get('identifier_column', '')}",
                f"Churn Count: {scan_result.get('churn_count', 0)}",
                "",
                "Customers Predicted to Churn:",
            ]
            customers = list(scan_result.get("churn_customers", []))
            if customers:
                lines.extend(f"- {customer}" for customer in customers)
            else:
                lines.append("- None")
            response = HttpResponse("\n".join(lines), content_type="text/plain")
            response["Content-Disposition"] = f'attachment; filename="{base_filename}.txt"'
            return response

        tabular_rows = _build_database_scan_tabular_rows(scan_result)
        if export_format == "xlsx":
            output = BytesIO()
            pd.DataFrame(tabular_rows).to_excel(output, index=False, sheet_name="Database Scan")
            response = HttpResponse(
                output.getvalue(),
                content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            response["Content-Disposition"] = f'attachment; filename="{base_filename}.xlsx"'
            return response

        csv_buffer = StringIO()
        fieldnames = ["identifier_column", "churn_count", "churn_customer"]
        writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
        writer.writeheader()
        for row in tabular_rows:
            writer.writerow(row)
        response = HttpResponse(csv_buffer.getvalue(), content_type="text/csv")
        response["Content-Disposition"] = f'attachment; filename="{base_filename}.csv"'
        return response

    # Fallback export for non-database scan entries.
    if export_format == "json":
        payload = {
            "generated_at_utc": datetime.utcnow().isoformat() + "Z",
            "username": str(request.user.username),
            "entry": entry,
        }
        response = HttpResponse(json.dumps(payload, indent=2), content_type="application/json")
        response["Content-Disposition"] = f'attachment; filename="{base_filename}.json"'
        return response

    if export_format == "txt":
        lines = [
            f"History Entry Report for {request.user.username}",
            f"Generated (UTC): {datetime.utcnow().isoformat()}Z",
            f"Timestamp: {entry.get('timestamp_utc', '')}",
            f"Analysis Type: {analysis_type}",
            f"Analyzed Data: {json.dumps(entry.get('analyzed_data', {}), ensure_ascii=True)}",
            f"Result Data: {json.dumps(entry.get('result_data', {}), ensure_ascii=True)}",
        ]
        response = HttpResponse("\n".join(lines), content_type="text/plain")
        response["Content-Disposition"] = f'attachment; filename="{base_filename}.txt"'
        return response

    row = {
        "timestamp_utc": entry.get("timestamp_utc", ""),
        "analysis_type": analysis_type,
        "analyzed_data": json.dumps(entry.get("analyzed_data", {}), ensure_ascii=True),
        "result_data": json.dumps(entry.get("result_data", {}), ensure_ascii=True),
    }
    if export_format == "xlsx":
        output = BytesIO()
        pd.DataFrame([row]).to_excel(output, index=False, sheet_name="History Entry")
        response = HttpResponse(
            output.getvalue(),
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        response["Content-Disposition"] = f'attachment; filename="{base_filename}.xlsx"'
        return response

    csv_buffer = StringIO()
    fieldnames = ["timestamp_utc", "analysis_type", "analyzed_data", "result_data"]
    writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow(row)
    response = HttpResponse(csv_buffer.getvalue(), content_type="text/csv")
    response["Content-Disposition"] = f'attachment; filename="{base_filename}.csv"'
    return response


def database_scan_result_download(request):
    if not request.user.is_authenticated:
        return redirect("home")

    scan_result = request.session.get("latest_database_scan_result")
    if not isinstance(scan_result, dict):
        return HttpResponse(
            "No scan result available to download. Run a database scan first.",
            status=404,
            content_type="text/plain",
        )

    export_format = str(request.GET.get("format", "csv")).strip().lower()
    if export_format not in {"csv", "json", "txt", "xlsx"}:
        export_format = "csv"

    timestamp_tag = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_username = "".join(ch for ch in str(request.user.username) if ch.isalnum() or ch in {"_", "-"}) or "user"
    base_filename = f"database_scan_report_{safe_username}_{timestamp_tag}"

    churn_customers = list(scan_result.get("churn_customers", []))
    churn_details = list(scan_result.get("churn_details", []))
    identifier_column = str(scan_result.get("identifier_column", "identifier"))

    if export_format == "json":
        payload = {
            "generated_at_utc": datetime.utcnow().isoformat() + "Z",
            "username": str(request.user.username),
            "report": {
                "identifier_column": identifier_column,
                "churn_count": int(scan_result.get("churn_count", 0)),
                "churn_details": churn_details,
                "churn_customers": churn_customers,
            },
        }
        response = HttpResponse(json.dumps(payload, indent=2), content_type="application/json")
        response["Content-Disposition"] = f'attachment; filename="{base_filename}.json"'
        return response

    if export_format == "txt":
        lines = [
            f"Database Scan Report for {request.user.username}",
            f"Generated (UTC): {datetime.utcnow().isoformat()}Z",
            f"Identifier Column: {identifier_column}",
            f"Churn Count: {scan_result.get('churn_count', 0)}",
            "",
            "Customers Predicted to Churn:",
        ]
        if churn_details:
            for detail in churn_details:
                lines.append(f"Customer: {detail.get('customer', '')}")
                lines.append(f"Risk: {detail.get('risk', '')}")
                lines.append(f"Probability: {detail.get('probability_percent', 0)}%")
                lines.append("")
        elif churn_customers:
            lines.extend(f"- {customer}" for customer in churn_customers)
        else:
            lines.append("- None")

        response = HttpResponse("\n".join(lines), content_type="text/plain")
        response["Content-Disposition"] = f'attachment; filename="{base_filename}.txt"'
        return response

    tabular_rows = _build_database_scan_tabular_rows(scan_result)

    if export_format == "xlsx":
        output = BytesIO()
        pd.DataFrame(tabular_rows).to_excel(output, index=False, sheet_name="Database Scan")
        response = HttpResponse(
            output.getvalue(),
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        response["Content-Disposition"] = f'attachment; filename="{base_filename}.xlsx"'
        return response

    csv_buffer = StringIO()
    fieldnames = ["identifier_column", "churn_count", "churn_customer", "risk", "probability_percent"]
    writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in tabular_rows:
        writer.writerow(row)

    response = HttpResponse(csv_buffer.getvalue(), content_type="text/csv")
    response["Content-Disposition"] = f'attachment; filename="{base_filename}.csv"'
    return response


def single_user_scan(request):
    if not request.user.is_authenticated:
        return redirect("home")

    result = None
    error_message = None

    if request.method == "POST":
        form = ChurnPredictionForm(request.POST)
        if form.is_valid():
            payload = form.cleaned_data
            try:
                prediction, confidence = predict_churn(payload)
                try:
                    result_payload = {
                        "label": "YES" if prediction == 1 else "NO",
                        "confidence_percent": round(confidence * 100, 2),
                    }
                    log_prediction_event(
                        request.user,
                        analysis_type="single_scan",
                        analyzed_data=payload,
                        result_data=result_payload,
                    )
                except Exception:
                    # Prediction should still be returned even if Mongo logging fails.
                    pass
                result = {
                    "label": "YES" if prediction == 1 else "NO",
                    "confidence": round(confidence * 100, 2),
                    "action": (
                        "Send retention offer immediately."
                        if prediction == 1
                        else "Customer is likely stable."
                    ),
                }
            except Exception as exc:
                error_message = str(exc)
    else:
        form = ChurnPredictionForm()

    context = {
        "form": form,
        "result": result,
        "error_message": error_message,
    }
    return render(request, "predictor/index.html", context)


def database_scan(request):
    if not request.user.is_authenticated:
        return redirect("home")

    scan_result = None
    error_message = None
    train_message = None

    if request.method == "POST":
        if "train_model" in request.POST:
            dataset_form = DatasetUploadForm(request.POST, request.FILES)
            if dataset_form.is_valid():
                try:
                    saved_path = train_from_uploaded_dataset(
                        dataset_form.cleaned_data["dataset_file"]
                    )
                    train_message = (
                        "Model trained and saved successfully: "
                        f"{saved_path.name}"
                    )
                except Exception as exc:
                    error_message = str(exc)
            else:
                error_message = "Please upload a valid CSV file to train the model."
        elif "scan_database" in request.POST:
            dataset_form = DatasetUploadForm(request.POST, request.FILES)
            if dataset_form.is_valid():
                try:
                    scan_result = predict_churn_customers_from_dataset(
                        dataset_form.cleaned_data["dataset_file"]
                    )
                    request.session["latest_database_scan_result"] = {
                        "total_rows": int(scan_result.get("total_rows", 0)),
                        "scanned_rows": int(scan_result.get("scanned_rows", 0)),
                        "identifier_column": str(scan_result.get("identifier_column", "")),
                        "churn_count": int(scan_result.get("churn_count", 0)),
                        "churn_customers": list(scan_result.get("churn_customers", [])),
                        "churn_details": list(scan_result.get("churn_details", [])),
                    }
                    try:
                        uploaded_file = dataset_form.cleaned_data["dataset_file"]
                        log_prediction_event(
                            request.user,
                            analysis_type="database_scan",
                            analyzed_data={
                                "file_name": str(uploaded_file.name),
                                "total_rows": int(scan_result.get("total_rows", 0)),
                                "scanned_rows": int(scan_result.get("scanned_rows", 0)),
                                "identifier_column": str(scan_result.get("identifier_column", "")),
                            },
                            result_data={
                                "churn_count": int(scan_result.get("churn_count", 0)),
                                "identifier_column": str(scan_result.get("identifier_column", "")),
                                "churn_customers": list(scan_result.get("churn_customers", [])),
                                "churn_details": list(scan_result.get("churn_details", [])),
                            },
                        )
                    except Exception:
                        pass
                except Exception as exc:
                    error_message = str(exc)
            else:
                error_message = "Please upload a valid CSV file to scan for churn."
        else:
            dataset_form = DatasetUploadForm()
    else:
        dataset_form = DatasetUploadForm()

    return render(
        request,
        "predictor/database_scan.html",
        {
            "dataset_form": dataset_form,
            "scan_result": scan_result,
            "error_message": error_message,
            "train_message": train_message,
        },
    )
