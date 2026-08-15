from __future__ import annotations

from .core_shared import *
from .layer import Layer
from .document_mixins.persistence import PersistenceDocumentMixin
from .document_mixins.project_io import ProjectIoDocumentMixin
from .document_mixins.smart_objects import SmartObjectsDocumentMixin
from .document_mixins.vector_layers import VectorLayersDocumentMixin
from .document_mixins.layer_geometry import LayerGeometryDocumentMixin
from .document_mixins.selections import SelectionsDocumentMixin
from .document_mixins.masks import MasksDocumentMixin
from .document_mixins.transforms import TransformsDocumentMixin


@dataclass
class Document(PersistenceDocumentMixin, ProjectIoDocumentMixin, SmartObjectsDocumentMixin, VectorLayersDocumentMixin, LayerGeometryDocumentMixin, SelectionsDocumentMixin, MasksDocumentMixin, TransformsDocumentMixin):
    width: int
    height: int
    dpi: int = 300
    color_model: str = "RGBA"
    bit_depth: int = 8
    background: tuple[int, int, int, int] = (255, 255, 255, 255)
    layers: list[Layer] = field(default_factory=list)
    active_layer: int = 0
    path: str | None = None
    dirty: bool = False
    selection_mask: np.ndarray | None = None
    saved_selections: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
