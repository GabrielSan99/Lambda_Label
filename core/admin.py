from django.contrib import admin

from .models import Annotation, Project, ROIMeasurement, Sample


@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = ("name", "owner", "created")
    search_fields = ("name",)


@admin.register(Sample)
class SampleAdmin(admin.ModelAdmin):
    list_display = ("name", "project", "device", "status", "n_bands", "width", "height", "linked", "created")
    list_filter = ("device", "status", "linked")
    search_fields = ("name", "source_path")


admin.site.register(Annotation)
admin.site.register(ROIMeasurement)
