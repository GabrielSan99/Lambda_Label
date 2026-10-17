import os
import shutil

from django.contrib.auth import get_user_model
from django.db.models.signals import post_delete, post_migrate
from django.dispatch import receiver

from . import spectral
from .models import Project, Sample

DEFAULT_SUPERUSER = "admin"
DEFAULT_SUPERUSER_PASSWORD = "lambda123"


@receiver(post_migrate)
def create_default_superuser(sender, **kwargs):
    """Um banco novo nasce com um usuario capaz de logar (todo o app exige login)."""
    if sender.name != "core":
        return
    User = get_user_model()
    if User.objects.filter(is_superuser=True).exists():
        return
    User.objects.create_superuser(username=DEFAULT_SUPERUSER, email="", password=DEFAULT_SUPERUSER_PASSWORD)
    print(f"Superusuário padrão criado: {DEFAULT_SUPERUSER} / {DEFAULT_SUPERUSER_PASSWORD} "
          f"(troque a senha depois do primeiro login).")


@receiver(post_delete, sender=Sample)
def delete_sample_folder(sender, instance, **kwargs):
    """Apaga a pasta da amostra (cubo convertido, previews, copias). Arquivos
    VINCULADOS (modo link) ficam fora dela e nunca sao tocados."""
    if not instance.folder:
        return
    spectral.forget_cube(instance.id)
    root = os.path.normpath(spectral.samples_root())
    path = os.path.normpath(spectral.sample_dir(instance))
    if path.startswith(root + os.sep) and path != root:
        shutil.rmtree(path, ignore_errors=True)


@receiver(post_delete, sender=Project)
def delete_project_models(sender, instance, **kwargs):
    for f in (instance.model_seg, instance.model_det, instance.model_cls):
        if f:
            f.delete(save=False)
