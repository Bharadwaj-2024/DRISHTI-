"""project_settings URL Configuration — DRISHTI v2
"""
from django.urls import path
from . import views
from .views import about, index, predict_page, cuda_full

app_name = 'ml_app'
handler404 = views.handler404

urlpatterns = [
    path('', index, name='home'),
    path('about/', about, name='about'),
    path('predict/', predict_page, name='predict'),
    path('url-ingest/', views.url_ingest, name='url_ingest'),
    path('report/', views.report_page, name='report_page'),
    path('report/download/', views.download_report, name='download_report'),
    path('report/pdf/', views.download_pdf_report, name='download_pdf_report'),
    path('feedback/', views.feedback_page, name='feedback_page'),
    path('feedback/submit/', views.submit_feedback, name='submit_feedback'),
    path('fact-check/', views.issue_fact_check, name='fact_check'),
    path('stats/', views.stats_page, name='stats_page'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('dashboard/login/', views.dashboard_login, name='dashboard_login'),
    path('dashboard/logout/', views.dashboard_logout, name='dashboard_logout'),
    path('cuda_full/', cuda_full, name='cuda_full'),
]
