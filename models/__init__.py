# models/ -- Student (Inception-Mamba) and Teacher (ViT-B/16) encoder definitions.
from models.inception_mamba import InceptionMambaEncoder, MAMBA_AVAILABLE

try:
    from models.vit_teacher import FrozenViTTeacher
except ImportError:
    # timm may not be installed or may be incompatible with the current
    # PyTorch version.  FrozenViTTeacher will still be importable via
    # the full path: from models.vit_teacher import FrozenViTTeacher
    # once the environment is correctly configured.
    FrozenViTTeacher = None  # type: ignore[assignment,misc]
