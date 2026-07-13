from django.urls import path
from django.contrib.auth import views as auth_views

from .views import (
    analysis_history,
    analysis_history_download,
    analysis_history_entry_download,
    database_scan,
    database_scan_result_download,
    dashboard,
    home,
    register,
    single_user_scan,
)

urlpatterns = [
    path("register/", register, name="register"),
    path(
        "login/",
        auth_views.LoginView.as_view(template_name="registration/login.html"),
        name="login",
    ),
    path("logout/", auth_views.LogoutView.as_view(next_page="home"), name="logout"),
    path("single-scan/", single_user_scan, name="single_scan"),
    path("database-scan/", database_scan, name="database_scan"),
    path("database-scan/download/", database_scan_result_download, name="database_scan_result_download"),
    path("dashboard/", dashboard, name="dashboard"),
    path("history/", analysis_history, name="analysis_history"),
    path("history/download/", analysis_history_download, name="analysis_history_download"),
    path("history/download-entry/", analysis_history_entry_download, name="analysis_history_entry_download"),
    path("", home, name="home"),
]
