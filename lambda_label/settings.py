"""
Configuracoes do Lambda Label.

Software de anotacao (manual + automatica) e machine learning sobre imagens
espectrais ja coletadas por outros equipamentos -- VideometerLab (.hips),
cameras hiperespectrais com cabecalho ENVI (.hdr/.bil/.bip/.bsq, ex.: Resonon),
SpectralCrops (bandas NNNnm.png), imagens comuns e cubos .npy genericos.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# SECURITY WARNING: troque em producao (variavel de ambiente LAMBDA_SECRET_KEY).
SECRET_KEY = os.environ.get(
    "LAMBDA_SECRET_KEY",
    "django-insecure-lambda-label-dev-key-troque-em-producao-9c1f4b7e2a",
)
DEBUG = os.environ.get("LAMBDA_DEBUG", "1") == "1"
ALLOWED_HOSTS = ["*"]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "core",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "lambda_label.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "lambda_label.wsgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
]

LOGIN_URL = "login"
LOGIN_REDIRECT_URL = "index"
LOGOUT_REDIRECT_URL = "login"

LANGUAGE_CODE = "pt-br"
TIME_ZONE = "America/Sao_Paulo"
USE_I18N = True
USE_TZ = False

STATIC_URL = "static/"

# Tudo que o app grava: amostras importadas (cubo convertido, preview, meta.json),
# modelos enviados/treinados e a area temporaria dos uploads (_staging/).
MEDIA_URL = "/data/"
MEDIA_ROOT = Path(os.environ.get("LAMBDA_DATA_DIR", BASE_DIR / "data"))

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Upload de pasta inteira pelo navegador manda um campo por arquivo -- o padrao
# do Django (100) recusaria uma pasta do SpectralCrops com algumas aquisicoes.
DATA_UPLOAD_MAX_NUMBER_FILES = 20000
DATA_UPLOAD_MAX_NUMBER_FIELDS = 40000
# Cubos hiperespectrais passam de 1 GB: acima disso o Django ja grava o upload
# em arquivo temporario (nao em memoria), entao so o limite do corpo importa.
DATA_UPLOAD_MAX_MEMORY_SIZE = 50 * 1024 * 1024

# Importar direto de uma pasta do disco do servidor (sem upload). Faz sentido
# quando o app roda na propria maquina onde os dados estao -- e o jeito de
# trabalhar com cubos de varios GB sem duplicar nada (modo "vincular").
# So usuarios staff podem usar; desligue se o servidor ficar exposto na rede.
LAMBDA_ALLOW_LOCAL_IMPORT = os.environ.get("LAMBDA_ALLOW_LOCAL_IMPORT", "1") == "1"
