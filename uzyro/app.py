from .app_shared import *
from .brand import apply_window_branding
from .app_mixins.startup_settings import StartupSettingsMixin
from .app_mixins.startup_screen import StartupScreenMixin
from .app_mixins.document_workspace import DocumentWorkspaceMixin
from .app_mixins.shortcuts import ShortcutsMixin
from .app_mixins.menus_tools import MenusToolsMixin
from .app_mixins.colors_canvas import ColorsCanvasMixin
from .app_mixins.color_picker import ColorPickerMixin
from .app_mixins.commands import CommandsMixin
from .app_mixins.action_workflows import ActionWorkflowsMixin
from .app_mixins.rendering_view import RenderingViewMixin
from .app_mixins.object_properties import ObjectPropertiesMixin
from .app_mixins.direct_manipulation import DirectManipulationMixin
from .app_mixins.selection_move_preview import SelectionMovePreviewMixin
from .app_mixins.pointer_support import PointerSupportMixin
from .app_mixins.source_overlay import SourceOverlayMixin
from .app_mixins.pointer_events import PointerEventsMixin
from .app_mixins.painting import PaintingMixin
from .app_mixins.brush_settings import BrushSettingsMixin
from .app_mixins.patch_interaction import PatchInteractionMixin
from .app_mixins.text_gradient import TextGradientMixin
from .app_mixins.gradient_editor import GradientEditorMixin
from .app_mixins.crop_overlays import CropOverlaysMixin
from .app_mixins.select_mask import SelectMaskMixin
from .app_mixins.automatic_selection import AutomaticSelectionMixin
from .app_mixins.color_range import ColorRangeMixin
from .app_mixins.documents import DocumentsMixin
from .app_mixins.smart_files import SmartFilesMixin
from .app_mixins.psd_files import PSDFileMixin
from .app_mixins.text_layers import TextLayersMixin
from .app_mixins.free_transform import FreeTransformMixin
from .app_mixins.transform_workspace import TransformWorkspaceMixin
from .app_mixins.transform_commands import TransformCommandsMixin
from .app_mixins.layer_styles import LayerStylesMixin
from .app_mixins.destructive_filters import DestructiveFiltersMixin
from .app_mixins.filter_dialogs import FilterDialogsMixin
from .app_mixins.filter_masks import FilterMasksMixin
from .app_mixins.layer_masks import LayerMasksMixin
from .app_mixins.text_vector import TextVectorMixin
from .app_mixins.shape_vector import ShapeVectorMixin
from .app_mixins.path_editing import PathEditingMixin
from .app_mixins.boolean_shapes import BooleanShapesMixin
from .app_mixins.image_geometry import ImageGeometryMixin
from .app_mixins.generative_workspace import GenerativeWorkspaceMixin
from .app_mixins.generative_history import GenerativeHistoryMixin
from .app_mixins.model_workspace import ModelWorkspaceMixin
from .app_mixins.color_workspace import ColorWorkspaceMixin
from .app_mixins.print_spot_workspace import PrintSpotWorkspaceMixin
from .app_mixins.adjustments import AdjustmentsMixin
from .app_mixins.retouch import RetouchMixin
from .app_mixins.advanced_retouch import AdvancedRetouchMixin
from .app_mixins.content_aware import ContentAwareMixin
from .app_mixins.plugins_view import PluginsViewMixin


class UZYROApp(StartupSettingsMixin, StartupScreenMixin, DocumentWorkspaceMixin, ShortcutsMixin, MenusToolsMixin, ColorsCanvasMixin, ColorPickerMixin, CommandsMixin, ActionWorkflowsMixin, RenderingViewMixin, ObjectPropertiesMixin, SelectionMovePreviewMixin, DirectManipulationMixin, PointerSupportMixin, SourceOverlayMixin, PathEditingMixin, PointerEventsMixin, PaintingMixin, BrushSettingsMixin, PatchInteractionMixin, GradientEditorMixin, TextGradientMixin, CropOverlaysMixin, SelectMaskMixin, ColorRangeMixin, AutomaticSelectionMixin, DocumentsMixin, PSDFileMixin, SmartFilesMixin, TextLayersMixin, FreeTransformMixin, TransformWorkspaceMixin, TransformCommandsMixin, LayerStylesMixin, FilterDialogsMixin, FilterMasksMixin, LayerMasksMixin, TextVectorMixin, ShapeVectorMixin, BooleanShapesMixin, ModelWorkspaceMixin, GenerativeHistoryMixin, GenerativeWorkspaceMixin, ImageGeometryMixin, ColorWorkspaceMixin, PrintSpotWorkspaceMixin, AdjustmentsMixin, DestructiveFiltersMixin, RetouchMixin, AdvancedRetouchMixin, ContentAwareMixin, PluginsViewMixin, tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        apply_window_branding(self)


def enable_high_dpi() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass

def main() -> None:
    enable_high_dpi()
    app = UZYROApp()
    if len(sys.argv) >= 3 and sys.argv[1] == "--smart-edit":
        project_path = Path(sys.argv[2]).resolve()
        try:
            app.open_document_session(Document.open_project(project_path), replace_startup=True)
            app.title(f"Содержимое Smart Object - {project_path.stem}")

            def save_nested_and_close() -> None:
                app.doc.save_project(project_path)
                app.destroy()

            app.protocol("WM_DELETE_WINDOW", save_nested_and_close)
        except Exception as exc:
            messagebox.showerror("Smart Object", f"Не удалось открыть вложенный документ:\n{exc}")
    app.mainloop()
