from .app_shared import *
from .app_mixins.startup_settings import StartupSettingsMixin
from .app_mixins.startup_screen import StartupScreenMixin
from .app_mixins.shortcuts import ShortcutsMixin
from .app_mixins.menus_tools import MenusToolsMixin
from .app_mixins.colors_canvas import ColorsCanvasMixin
from .app_mixins.commands import CommandsMixin
from .app_mixins.rendering_view import RenderingViewMixin
from .app_mixins.pointer_support import PointerSupportMixin
from .app_mixins.pointer_events import PointerEventsMixin
from .app_mixins.painting import PaintingMixin
from .app_mixins.text_gradient import TextGradientMixin
from .app_mixins.crop_overlays import CropOverlaysMixin
from .app_mixins.select_mask import SelectMaskMixin
from .app_mixins.automatic_selection import AutomaticSelectionMixin
from .app_mixins.documents import DocumentsMixin
from .app_mixins.smart_files import SmartFilesMixin
from .app_mixins.text_layers import TextLayersMixin
from .app_mixins.free_transform import FreeTransformMixin
from .app_mixins.transform_workspace import TransformWorkspaceMixin
from .app_mixins.filter_dialogs import FilterDialogsMixin
from .app_mixins.filter_masks import FilterMasksMixin
from .app_mixins.layer_masks import LayerMasksMixin
from .app_mixins.text_vector import TextVectorMixin
from .app_mixins.shape_vector import ShapeVectorMixin
from .app_mixins.boolean_shapes import BooleanShapesMixin
from .app_mixins.image_geometry import ImageGeometryMixin
from .app_mixins.adjustments import AdjustmentsMixin
from .app_mixins.retouch import RetouchMixin
from .app_mixins.content_aware import ContentAwareMixin
from .app_mixins.plugins_view import PluginsViewMixin


class PhotoRedactorApp(StartupSettingsMixin, StartupScreenMixin, ShortcutsMixin, MenusToolsMixin, ColorsCanvasMixin, CommandsMixin, RenderingViewMixin, PointerSupportMixin, PointerEventsMixin, PaintingMixin, TextGradientMixin, CropOverlaysMixin, SelectMaskMixin, AutomaticSelectionMixin, DocumentsMixin, SmartFilesMixin, TextLayersMixin, FreeTransformMixin, TransformWorkspaceMixin, FilterDialogsMixin, FilterMasksMixin, LayerMasksMixin, TextVectorMixin, ShapeVectorMixin, BooleanShapesMixin, ImageGeometryMixin, AdjustmentsMixin, RetouchMixin, ContentAwareMixin, PluginsViewMixin, tk.Tk):
    pass


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
    app = PhotoRedactorApp()
    if len(sys.argv) >= 3 and sys.argv[1] == "--smart-edit":
        project_path = Path(sys.argv[2]).resolve()
        try:
            app.doc = Document.open_project(project_path)
            app._edit_generation += 1
            app.history.clear()
            app.selection_box = app.doc.selection_bounds()
            app.show_editor()
            app.title(f"Содержимое Smart Object - {project_path.stem}")

            def save_nested_and_close() -> None:
                app.doc.save_project(project_path)
                app.destroy()

            app.protocol("WM_DELETE_WINDOW", save_nested_and_close)
        except Exception as exc:
            messagebox.showerror("Smart Object", f"Не удалось открыть вложенный документ:\n{exc}")
    app.mainloop()
