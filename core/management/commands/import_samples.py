"""Importa amostras pela linha de comando -- pra automatizar a entrada de
dados de qualquer equipamento (ex.: um script que roda depois de cada coleta).

    python manage.py import_samples "Meu projeto" D:\\coletas\\2026-10-17 --link
    python manage.py import_samples 3 arquivo.hips outro.bil.hdr --owner admin
"""
from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError

from core import importers
from core.models import Project, Sample


class Command(BaseCommand):
    help = "Importa imagens (.hips, ENVI .hdr, .npy, pastas do SpectralCrops, imagens) para um projeto."

    def add_arguments(self, parser):
        parser.add_argument("project", help="id ou nome do projeto (criado se não existir, com --owner)")
        parser.add_argument("paths", nargs="+", help="arquivos ou pastas")
        parser.add_argument("--owner", default=None, help="usuário dono ao criar o projeto (padrão: 1º superusuário)")
        parser.add_argument("--link", action="store_true",
                            help="ler cubos ENVI/.npy direto do original, sem copiar")
        parser.add_argument("--no-group", action="store_true", help="não agrupar imagens com o mesmo prefixo")
        parser.add_argument("--dry-run", action="store_true", help="só listar o que seria importado")

    def handle(self, *args, **opts):
        project = self._project(opts["project"], opts["owner"])
        items, ignored = importers.scan(opts["paths"], group_by_prefix=not opts["no_group"])
        self.stdout.write(f"{len(items)} item(ns) reconhecido(s), {len(ignored)} arquivo(s) ignorado(s).")
        for it in items:
            flag = f"  ERRO: {it.error}" if it.error else ""
            self.stdout.write(f"  [{it.device}] {it.name} — {it.info}{flag}")
        if opts["dry_run"]:
            return
        ok = err = 0
        for it in items:
            if it.error:
                err += 1
                continue
            s = importers.import_item(project, it, link=opts["link"])
            if s.status == Sample.Status.READY:
                ok += 1
            else:
                err += 1
                self.stderr.write(f"  falhou: {s.name}: {s.error}")
        self.stdout.write(self.style.SUCCESS(f"Importadas {ok} amostra(s) em “{project.name}”.")
                          + (f" {err} com erro." if err else ""))

    def _project(self, ref, owner):
        if ref.isdigit() and Project.objects.filter(id=int(ref)).exists():
            return Project.objects.get(id=int(ref))
        qs = Project.objects.filter(name=ref)
        if qs.count() == 1:
            return qs.first()
        User = get_user_model()
        user = (User.objects.filter(username=owner).first() if owner
                else User.objects.filter(is_superuser=True).order_by("id").first())
        if not user:
            raise CommandError("Usuário dono não encontrado (use --owner).")
        return Project.objects.get_or_create(owner=user, name=ref)[0]
