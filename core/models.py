from django.conf import settings
from django.db import models


class Project(models.Model):
    """Agrupa amostras importadas pra anotar, medir e treinar modelos.

    Diferente do SpectralCrops (onde o projeto tambem guardava a parametrizacao
    da camera/LEDs), aqui o projeto nao conhece hardware nenhum: as imagens
    chegam prontas de outro equipamento, cada amostra traz o proprio numero de
    bandas, comprimentos de onda e escala (ver Sample)."""
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="projects")
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True, default="")
    # mm por pixel usado nas amostras que nao trazem essa informacao no arquivo
    # (o .hips do Videometer traz; ENVI/imagens normalmente nao). Vazio = so px.
    mm_per_pixel = models.FloatField(null=True, blank=True)

    model_seg = models.FileField(upload_to="models/", blank=True)  # YOLO segmentacao (.pt)
    model_det = models.FileField(upload_to="models/", blank=True)  # YOLO deteccao (.pt)
    # Classificacao: ou um YOLO-cls (.pt, classifica o recorte RGB) ou o
    # classificador espectral treinado na pagina Machine Learning (.joblib).
    model_cls = models.FileField(upload_to="models/", blank=True)
    # Metadados do ultimo treino espectral -- ver ml.train_classifier.
    model_cls_info = models.JSONField(default=dict, blank=True)
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["name"]
        unique_together = [("owner", "name")]

    def __str__(self):
        return self.name


class Sample(models.Model):
    """Uma imagem espectral importada (um cubo H x W x bandas).

    O cubo fica onde for mais barato ler: cubos ENVI podem ser lidos direto do
    arquivo original via memmap (linked=True, nada e copiado -- um .bil do
    Resonon tem ~1 GB); os outros formatos sao convertidos uma vez pra
    cube.npy dentro da pasta da amostra. `storage` diz como abrir (ver
    spectral.open_cube). Valor armazenado * value_scale = fracao 0..1
    (reflectancia, ou intensidade relativa quando unit="intensity")."""

    class Device(models.TextChoices):
        VIDEOMETER = "videometer", "VideometerLab (.hips)"
        ENVI = "envi", "Hiperespectral ENVI (.hdr)"
        SPECTRALCROPS = "spectralcrops", "SpectralCrops (bandas PNG)"
        IMAGE = "image", "Imagem / conjunto de imagens"
        NPY = "npy", "Cubo NumPy (.npy)"

    class Status(models.TextChoices):
        PENDING = "pending", "Importando"
        READY = "ready", "Pronta"
        ERROR = "error", "Erro"

    class Unit(models.TextChoices):
        REFLECTANCE = "reflectance", "Reflectância"
        INTENSITY = "intensity", "Intensidade relativa"

    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name="samples")
    name = models.CharField(max_length=300)
    device = models.CharField(max_length=20, choices=Device.choices)
    status = models.CharField(max_length=10, choices=Status.choices, default=Status.PENDING)
    error = models.TextField(blank=True, default="")

    # pasta da amostra, relativa a MEDIA_ROOT/samples (preview, thumb, meta.json, cube.npy)
    folder = models.CharField(max_length=300, blank=True)
    linked = models.BooleanField(default=False)          # cubo lido do arquivo original (nao copiado)
    source_path = models.CharField(max_length=2000, blank=True)  # de onde veio (informativo)
    storage = models.JSONField(default=dict, blank=True)

    height = models.PositiveIntegerField(default=0)
    width = models.PositiveIntegerField(default=0)
    n_bands = models.PositiveIntegerField(default=0)
    wavelengths = models.JSONField(default=list, blank=True)   # [nm, ...] ou [] se o formato nao tem
    band_names = models.JSONField(default=list, blank=True)
    value_scale = models.FloatField(default=1.0)
    unit = models.CharField(max_length=12, choices=Unit.choices, default=Unit.REFLECTANCE)
    mm_per_pixel = models.FloatField(null=True, blank=True)
    rgb_bands = models.JSONField(default=list, blank=True)     # [r, g, b] indices usados no preview
    meta = models.JSONField(default=dict, blank=True)          # resto do cabecalho original
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["name", "id"]

    def __str__(self):
        return self.name

    DEVICE_SHORT = {"videometer": "Videometer", "envi": "ENVI", "spectralcrops": "SpectralCrops",
                    "image": "Imagem", "npy": "NumPy"}

    @property
    def device_short(self):
        return self.DEVICE_SHORT.get(self.device, self.device)

    @property
    def effective_mm_per_pixel(self):
        return self.mm_per_pixel or self.project.mm_per_pixel

    def band_label(self, i):
        """Rotulo curto da banda i: '660 nm' quando ha comprimento de onda, senao o nome."""
        wl = self.wavelengths[i] if i < len(self.wavelengths) else None
        if wl is not None:
            return f"{wl:g} nm"
        if i < len(self.band_names) and self.band_names[i]:
            return self.band_names[i]
        return f"banda {i + 1}"


class Annotation(models.Model):
    """Anotacao de uma amostra -- mesma estrutura do SpectralCrops: cada tipo
    de forma no seu proprio campo (elipse nunca mistura com poligono)."""
    sample = models.OneToOneField(Sample, on_delete=models.CASCADE, related_name="annotation")
    boxes = models.JSONField(default=list)          # [[x1,y1,x2,y2], ...]
    labels = models.JSONField(default=list)
    ellipses = models.JSONField(default=list)       # [[[x,y], ...], ...] (aproximacao poligonal)
    ellipse_labels = models.JSONField(default=list)
    polygons = models.JSONField(default=list)       # [[[x,y], ...], ...]
    poly_labels = models.JSONField(default=list)
    points = models.JSONField(default=list)         # [[x,y], ...]
    point_labels = models.JSONField(default=list)
    image_label = models.CharField(max_length=200, blank=True)  # classe/condicao da amostra inteira
    image_note = models.TextField(blank=True, default="")
    updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Anotação de {self.sample}"

    @property
    def n_shapes(self):
        return len(self.boxes) + len(self.ellipses) + len(self.polygons) + len(self.points)


class ROIMeasurement(models.Model):
    """Espectro medio (e desvio) de UM ROI desenhado numa amostra. Os valores
    seguem a ordem de bandas da propria amostra (sample.wavelengths), em %
    (0..100 = fracao 0..1 do cubo). Area/caixa em px sempre; em mm quando ha
    mm/pixel (do arquivo ou do projeto)."""

    class Method(models.TextChoices):
        BOX = "box", "Retângulo"
        ELLIPSE = "ellipse", "Elipse"
        POLYGON = "polygon", "Polígono"
        POINT = "point", "Ponto"

    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name="roi_measurements")
    sample = models.ForeignKey(Sample, on_delete=models.CASCADE, related_name="roi_measurements")
    method = models.CharField(max_length=12, choices=Method.choices)
    index = models.PositiveIntegerField(default=0)      # ordem do ROI dentro do metodo (1..n)
    label = models.CharField(max_length=200, blank=True)
    contour = models.JSONField(default=dict, blank=True)
    spectrum = models.JSONField(default=list)           # media por banda (%), None onde nao ha pixel valido
    spectrum_std = models.JSONField(default=list)
    n_pixels = models.PositiveIntegerField(default=0)
    area_px = models.FloatField(null=True, blank=True)
    area_mm2 = models.FloatField(null=True, blank=True)
    bbox_width_px = models.FloatField(null=True, blank=True)
    bbox_height_px = models.FloatField(null=True, blank=True)
    bbox_width_mm = models.FloatField(null=True, blank=True)
    bbox_height_mm = models.FloatField(null=True, blank=True)
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["sample", "method", "index"]
        unique_together = [("sample", "method", "index")]

    def __str__(self):
        return f"ROI {self.index} ({self.method}) de {self.sample}"
