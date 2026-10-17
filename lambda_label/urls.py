from django.contrib import admin
from django.urls import include, path

from core import views

urlpatterns = [
    path("accounts/", include("django.contrib.auth.urls")),
    path("admin/", admin.site.urls),
    path("", views.index, name="index"),

    # Projetos
    path("projects/create/", views.project_create, name="project_create"),
    path("projects/<int:pid>/", views.project_view, name="project"),
    path("projects/<int:pid>/update/", views.project_update, name="project_update"),
    path("projects/<int:pid>/delete/", views.project_delete, name="project_delete"),
    path("projects/<int:pid>/rois/recompute/", views.project_rois_recompute, name="project_rois_recompute"),

    # Importacao de imagens coletadas em outros equipamentos
    path("projects/<int:pid>/import/scan/", views.import_scan, name="import_scan"),
    path("projects/<int:pid>/import/run/", views.import_run, name="import_run"),

    # Exportacoes (seg | det | coco | rois | samples)
    path("projects/<int:pid>/export/<str:kind>/", views.project_export, name="project_export"),

    # Analise e Machine Learning
    path("projects/<int:pid>/analysis/", views.analysis, name="analysis"),
    path("projects/<int:pid>/ml/", views.ml_page, name="ml"),
    path("projects/<int:pid>/ml/estimate/", views.ml_estimate, name="ml_estimate"),
    path("projects/<int:pid>/ml/train/", views.ml_train, name="ml_train"),
    path("projects/<int:pid>/ml/model/", views.ml_model_upload, name="ml_model_upload"),
    path("projects/<int:pid>/ml/model/delete/", views.ml_model_delete, name="ml_model_delete"),
    path("ml/kit/", views.ml_kit, name="ml_kit"),

    # Tarefas em segundo plano
    path("jobs/<str:job_id>/", views.job_status, name="job_status"),
    path("jobs/<str:job_id>/download/", views.job_download, name="job_download"),

    # Amostra: editor de anotacao, imagens e auto-anotacao
    path("samples/<int:sid>/", views.annotate, name="annotate"),
    path("samples/<int:sid>/save/", views.annotation_save, name="annotation_save"),
    path("samples/<int:sid>/label/", views.annotation_image_label, name="annotation_image_label"),
    path("samples/<int:sid>/note/", views.annotation_image_note, name="annotation_image_note"),
    path("samples/<int:sid>/update/", views.sample_update, name="sample_update"),
    path("samples/<int:sid>/delete/", views.sample_delete, name="sample_delete"),
    path("samples/<int:sid>/rois/compute/", views.roi_compute, name="roi_compute"),
    path("samples/<int:sid>/spectrum/", views.roi_spectrum, name="roi_spectrum"),
    path("samples/<int:sid>/auto/<str:method>/", views.auto_annotate, name="auto_annotate"),
    path("samples/<int:sid>/preview.png", views.sample_preview, name="sample_preview"),
    path("samples/<int:sid>/thumb.jpg", views.sample_thumb, name="sample_thumb"),
    path("samples/<int:sid>/band/<int:i>.png", views.sample_band, name="sample_band"),
    path("samples/<int:sid>/composite.png", views.sample_composite, name="sample_composite"),
]
