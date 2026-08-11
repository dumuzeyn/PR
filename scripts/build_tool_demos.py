from __future__ import annotations

import math
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "photoredactor" / "assets" / "tool_demos"
SOURCE_DIR = ROOT / "design_assets" / "tool_demo_sources"
SIZE = (288, 162)
FRAME_COUNT = 18

TOOLS = (
    "hand", "move", "brush", "eraser", "blur_tool", "sharpen_tool", "dodge", "burn",
    "clone", "healing", "spot_healing", "patch", "fill", "gradient", "text", "eyedropper",
    "rect_shape", "ellipse_shape", "line_shape", "bezier_shape", "polygon_shape", "star_shape",
    "path_select", "direct_select", "add_anchor", "delete_anchor", "convert_anchor",
    "custom_shape", "select", "ellipse_select", "lasso", "magnetic_lasso", "polygon_lasso",
    "quick_selection", "magic_wand", "color_range", "crop",
)

ACTION_LABELS = {
    "hand": "ПЕРЕМЕЩЕНИЕ ХОЛСТА", "move": "ПЕРЕМЕЩЕНИЕ СЛОЯ", "brush": "РИСОВАНИЕ",
    "eraser": "СТИРАНИЕ", "blur_tool": "РАЗМЫТИЕ", "sharpen_tool": "УСИЛЕНИЕ ДЕТАЛЕЙ",
    "dodge": "ОСВЕТЛЕНИЕ", "burn": "ЗАТЕМНЕНИЕ", "clone": "КОПИРОВАНИЕ ИЗ ИСТОЧНИКА",
    "healing": "ЛЕЧЕНИЕ С УЧЕТОМ ФОНА", "spot_healing": "УДАЛЕНИЕ ДЕФЕКТА",
    "patch": "ПЕРЕНОС ЧИСТОГО УЧАСТКА", "fill": "ЗАЛИВКА ОБЛАСТИ", "gradient": "РАСТЯГИВАНИЕ ПЕРЕХОДА",
    "text": "ВВОД ТЕКСТА", "eyedropper": "ВЫБОР ЦВЕТА", "rect_shape": "СОЗДАНИЕ ПРЯМОУГОЛЬНИКА",
    "ellipse_shape": "СОЗДАНИЕ ЭЛЛИПСА", "line_shape": "СОЗДАНИЕ ЛИНИИ", "bezier_shape": "ИЗГИБ КРИВОЙ",
    "path_select": "ВЫБОР КОНТУРА", "direct_select": "ПЕРЕМЕЩЕНИЕ УЗЛА",
    "add_anchor": "ДОБАВЛЕНИЕ УЗЛА", "delete_anchor": "УДАЛЕНИЕ УЗЛА",
    "convert_anchor": "УГЛОВОЙ И ПЛАВНЫЙ УЗЕЛ",
    "polygon_shape": "СОЗДАНИЕ МНОГОУГОЛЬНИКА", "star_shape": "СОЗДАНИЕ ЗВЕЗДЫ",
    "custom_shape": "СОЗДАНИЕ СВОЕЙ ФИГУРЫ", "select": "ПРЯМОУГОЛЬНОЕ ВЫДЕЛЕНИЕ",
    "ellipse_select": "ОВАЛЬНОЕ ВЫДЕЛЕНИЕ", "lasso": "СВОБОДНЫЙ КОНТУР",
    "magnetic_lasso": "ПРИВЯЗКА К КРАЮ", "polygon_lasso": "КОНТУР ПО ВЕРШИНАМ",
    "quick_selection": "РАСШИРЕНИЕ ВЫДЕЛЕНИЯ", "magic_wand": "ПОХОЖАЯ СВЯЗАННАЯ ОБЛАСТЬ",
    "color_range": "ОДИН ЦВЕТ ПО ВСЕМУ КАДРУ", "crop": "ОБРЕЗКА КАДРА",
}


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "segoeuib.ttf" if bold else "segoeui.ttf"
    path = Path("C:/Windows/Fonts") / name
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default()


def cover(path: Path, size: tuple[int, int] = SIZE) -> Image.Image:
    return ImageOps.fit(Image.open(path).convert("RGB"), size, Image.Resampling.LANCZOS)


def checker(size: tuple[int, int] = SIZE, cell: int = 9) -> Image.Image:
    image = Image.new("RGB", size, "#d9d9d9")
    draw = ImageDraw.Draw(image)
    for y in range(0, size[1], cell):
        for x in range(0, size[0], cell):
            if (x // cell + y // cell) % 2:
                draw.rectangle((x, y, x + cell - 1, y + cell - 1), fill="#bdbdbd")
    return image


def cursor(image: Image.Image, point: tuple[float, float], *, ring: int = 0) -> None:
    x, y = int(point[0]), int(point[1])
    draw = ImageDraw.Draw(image)
    if ring:
        draw.ellipse((x - ring, y - ring, x + ring, y + ring), outline="#ffffff", width=2)
        draw.ellipse((x - ring - 1, y - ring - 1, x + ring + 1, y + ring + 1), outline="#1b1b1b")
    points = [(x, y), (x + 3, y + 17), (x + 7, y + 12), (x + 12, y + 19), (x + 16, y + 16), (x + 10, y + 9), (x + 17, y + 7)]
    draw.polygon(points, fill="#ffffff", outline="#111111")


def lerp(a: tuple[float, float], b: tuple[float, float], t: float) -> tuple[float, float]:
    return a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t


def dashed_line(draw: ImageDraw.ImageDraw, points: list[tuple[int, int]], phase: int = 0, width: int = 2) -> None:
    for start, end in zip(points, points[1:]):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = max(1, int(math.hypot(dx, dy)))
        for pos in range(-phase, length, 8):
            lo, hi = max(0, pos), min(length, pos + 4)
            if hi <= lo:
                continue
            p1 = (start[0] + dx * lo / length, start[1] + dy * lo / length)
            p2 = (start[0] + dx * hi / length, start[1] + dy * hi / length)
            draw.line((p1, p2), fill="#ffffff", width=width + 1)
            draw.line((p1, p2), fill="#151515", width=width)


def closed_dashes(draw: ImageDraw.ImageDraw, points: list[tuple[int, int]], phase: int) -> None:
    dashed_line(draw, points + [points[0]], phase)


def composite_effect(base: Image.Image, effect: Image.Image, mask: Image.Image) -> Image.Image:
    return Image.composite(effect, base, mask.convert("L"))


def ellipse_mask(box: tuple[int, int, int, int], amount: float = 1.0) -> Image.Image:
    mask = Image.new("L", SIZE, 0)
    if amount > 0:
        x0, y0, x1, y1 = box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        rx, ry = (x1 - x0) * amount / 2, (y1 - y0) * amount / 2
        ImageDraw.Draw(mask).ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=255)
    return mask


def progress(index: int) -> float:
    if index < 3:
        return 0.0
    if index >= FRAME_COUNT - 4:
        return 1.0
    return (index - 3) / (FRAME_COUNT - 8)


def stage_banner(image: Image.Image, tool: str, index: int) -> None:
    if index < 3:
        text = "ДО"
        color = "#505761"
    elif index >= FRAME_COUNT - 4:
        text = "РЕЗУЛЬТАТ"
        color = "#167447"
    else:
        text = ACTION_LABELS[tool]
        color = "#1769aa"
    draw = ImageDraw.Draw(image, "RGBA")
    label_font = font(10, True)
    bounds = draw.textbbox((0, 0), text, font=label_font)
    width = bounds[2] - bounds[0] + 14
    draw.rounded_rectangle((6, 6, 6 + width, 25), radius=3, fill=color + "E8")
    draw.text((13, 8), text, font=label_font, fill="#ffffff")


def draw_shape(image: Image.Image, tool: str, t: float, phase: int) -> tuple[float, float]:
    draw = ImageDraw.Draw(image, "RGBA")
    start, end = (74, 42), (222, 128)
    ex, ey = lerp(start, end, t)
    box = (start[0], start[1], int(ex), int(ey))
    fill, outline = (20, 132, 220, 95), (255, 255, 255, 255)
    if tool == "rect_shape":
        draw.rectangle(box, fill=fill, outline=outline, width=3)
    elif tool == "ellipse_shape":
        draw.ellipse(box, fill=fill, outline=outline, width=3)
    elif tool == "line_shape":
        draw.line((start, (ex, ey)), fill=outline, width=5)
    elif tool == "bezier_shape":
        points = []
        for step in range(31):
            u = step / 30
            x = (1-u)**3*start[0] + 3*(1-u)**2*u*95 + 3*(1-u)*u*u*205 + u**3*ex
            y = (1-u)**3*start[1] + 3*(1-u)**2*u*145 + 3*(1-u)*u*u*25 + u**3*ey
            points.append((x, y))
        draw.line(points, fill=outline, width=4)
        draw.line((start, (95, 145)), fill=(255, 255, 255, 150), width=1)
        draw.line(((205, 25), (ex, ey)), fill=(255, 255, 255, 150), width=1)
        for p in ((95, 145), (205, 25)):
            draw.ellipse((p[0]-3, p[1]-3, p[0]+3, p[1]+3), fill="#ffffff")
    elif tool == "polygon_shape":
        pts = [(144, 38), (int(ex), 82), (185, int(ey)), (102, int(ey)), (74, 82)]
        draw.polygon(pts, fill=fill, outline=outline)
    elif tool == "star_shape":
        pts = []
        cx, cy, radius = 148, 84, 54 * t
        for n in range(10):
            angle = -math.pi / 2 + n * math.pi / 5
            r = radius if n % 2 == 0 else radius * 0.43
            pts.append((cx + math.cos(angle) * r, cy + math.sin(angle) * r))
        draw.polygon(pts, fill=fill, outline=outline)
        ex, ey = pts[2] if pts else start
    else:
        cx, cy, s = 148, 82, 55 * t
        pts = [(cx, cy+s), (cx-s, cy-4), (cx-s*.78, cy-s*.72), (cx-s*.35, cy-s*.78), (cx, cy-s*.32), (cx+s*.35, cy-s*.78), (cx+s*.78, cy-s*.72), (cx+s, cy-4)]
        draw.polygon(pts, fill=fill, outline=outline)
        ex, ey = cx + s, cy
    return ex, ey


def selection_demo(base: Image.Image, tool: str, t: float, phase: int) -> tuple[Image.Image, tuple[float, float]]:
    image = base.copy()
    draw = ImageDraw.Draw(image, "RGBA")
    if tool == "select":
        start, end = (72, 39), lerp((72, 39), (225, 132), t)
        pts = [(72, 39), (int(end[0]), 39), (int(end[0]), int(end[1])), (72, int(end[1]))]
        closed_dashes(draw, pts, phase)
        point = end
    elif tool == "ellipse_select":
        box = (70, 35, int(70 + 158*t), int(35 + 100*t))
        draw.ellipse(box, fill=(38, 132, 220, 35), outline="#ffffff", width=2)
        point = (box[2], box[3])
    elif tool in {"lasso", "magnetic_lasso"}:
        contour = [(22, 113), (27, 91), (37, 79), (52, 77), (65, 86), (72, 107), (65, 129), (47, 137), (29, 129)]
        count = max(2, int(2 + t * (len(contour) - 1)))
        used = contour[:count]
        if t >= 1:
            closed_dashes(draw, contour, phase)
        else:
            dashed_line(draw, used, phase)
        if tool == "magnetic_lasso":
            for x, y in used[::2]:
                draw.ellipse((x-2, y-2, x+2, y+2), fill="#2d8fdd", outline="#ffffff")
        point = used[-1]
    elif tool == "polygon_lasso":
        contour = [(72, 126), (91, 55), (178, 36), (232, 92), (194, 137)]
        count = max(2, int(2 + t * (len(contour) - 1)))
        used = contour[:count]
        dashed_line(draw, used, phase)
        for x, y in used:
            draw.rectangle((x-2, y-2, x+2, y+2), fill="#ffffff", outline="#111111")
        if t >= 1:
            closed_dashes(draw, contour, phase)
        point = used[-1]
    elif tool == "quick_selection":
        box = (17, 76, int(42 + 34*t), int(105 + 36*t))
        draw.ellipse(box, fill=(38, 132, 220, 70), outline="#ffffff", width=2)
        point = (36 + 25*t, 108)
    elif tool == "magic_wand":
        alpha = int(120*t)
        draw.rectangle((0, 0, 288, 88), fill=(38, 132, 220, alpha))
        dashed_line(draw, [(0, 88), (288, 88)], phase)
        point = (205, 42)
    else:
        alpha = int(90*t)
        draw.ellipse((16, 76, 75, 141), fill=(226, 52, 52, alpha), outline="#ffffff", width=2)
        draw.ellipse((221, 119, 239, 139), fill=(226, 52, 52, alpha), outline="#ffffff", width=2)
        point = (46, 103)
    cursor(image, point, ring=10 if tool == "quick_selection" else 0)
    return image, point


def make_frames(tool: str, still: Image.Image, landscape: Image.Image) -> list[Image.Image]:
    frames: list[Image.Image] = []
    for index in range(FRAME_COUNT):
        t = progress(index)
        phase = index % 8
        landscape_tools = {"hand", "crop", "select", "ellipse_select", "polygon_lasso", "magic_wand"}
        base = landscape.copy() if tool in landscape_tools else still.copy()
        if tool == "color_range":
            base_draw = ImageDraw.Draw(base)
            base_draw.ellipse((221, 120, 231, 133), fill="#d72e25")
            base_draw.ellipse((229, 116, 240, 131), fill="#c92524")

        if tool == "hand":
            large = ImageOps.fit(landscape, (332, 186), Image.Resampling.LANCZOS)
            offset = int(20 * t)
            image = large.crop((22 - offset, 12, 310 - offset, 174))
            ImageDraw.Draw(image).line((205, 84, 132, 84), fill="#ffffff", width=3)
            cursor(image, lerp((205, 84), (130, 84), t), ring=9)
        elif tool == "move":
            image = base.copy()
            source = still.crop((16, 75, 76, 143))
            mask = Image.new("L", source.size, 0)
            ImageDraw.Draw(mask).ellipse((2, 2, 58, 66), fill=255)
            x = int(16 + 120*t)
            image.paste(checker((60, 68)), (16, 75), mask if t > 0 else None)
            image.paste(source, (x, 75), mask)
            ImageDraw.Draw(image).rectangle((x, 75, x+59, 142), outline="#ffffff", width=2)
            cursor(image, (x+38, 108))
        elif tool in {"brush", "eraser"}:
            image = base.copy()
            path = [(52 + int(150*s), 115 - int(55*math.sin(s*math.pi))) for s in [n/20 for n in range(21)]]
            count = max(2, int(2 + t*(len(path)-2)))
            if tool == "brush":
                ImageDraw.Draw(image).line(path[:count], fill="#e83232", width=18, joint="curve")
            else:
                mask = Image.new("L", SIZE, 0)
                ImageDraw.Draw(mask).line(path[:count], fill=255, width=22, joint="curve")
                image = Image.composite(checker(), image, mask)
            cursor(image, path[count-1], ring=11)
        elif tool in {"blur_tool", "sharpen_tool", "dodge", "burn"}:
            if tool == "blur_tool":
                effect = base.filter(ImageFilter.GaussianBlur(7))
            elif tool == "sharpen_tool":
                base = still.filter(ImageFilter.GaussianBlur(1.8))
                effect = ImageEnhance.Sharpness(still).enhance(3.2)
            elif tool == "dodge":
                effect = ImageEnhance.Brightness(base).enhance(1.65)
            else:
                effect = ImageEnhance.Brightness(base).enhance(0.48)
            box = (78, 78, 140, 141)
            image = composite_effect(base, effect, ellipse_mask(box, t))
            if tool in {"blur_tool", "sharpen_tool"}:
                detail = image.crop((82, 82, 136, 136)).resize((70, 70), Image.Resampling.NEAREST)
                image.paste(detail, (210, 48))
                detail_draw = ImageDraw.Draw(image)
                detail_draw.rectangle((208, 46, 281, 120), outline="#ffffff", width=3)
                detail_draw.line((139, 84, 208, 52), fill="#ffffff", width=2)
            cursor(image, lerp((94, 111), (126, 111), t), ring=18)
        elif tool == "clone":
            image = base.copy()
            source = still.crop((77, 86, 132, 139))
            mask = Image.new("L", source.size, 0)
            ImageDraw.Draw(mask).ellipse((0, 4, 54, 49), fill=255)
            draw = ImageDraw.Draw(image)
            draw.line((104, 93, 104, 113), fill="#ffffff", width=2)
            draw.line((94, 103, 114, 103), fill="#ffffff", width=2)
            draw.line((113, 103, 154, 103), fill="#ffffff", width=2)
            if t > .25:
                image.paste(source, (145, 88), mask)
                draw.ellipse((144, 87, 201, 141), outline="#ffffff", width=2)
            cursor(image, lerp((104, 103), (168, 109), t), ring=13)
        elif tool in {"healing", "spot_healing", "patch"}:
            damaged = base.copy()
            d = ImageDraw.Draw(damaged)
            spots = [(39, 103, 7), (53, 118, 6), (99, 108, 6)]
            for x, y, r in spots:
                d.ellipse((x-r, y-r, x+r, y+r), fill="#3b251d")
            if tool == "patch":
                d.line((151, 92, 174, 119), fill="#502b24", width=4)
                image = damaged if t < .72 else base.copy()
                draw = ImageDraw.Draw(image)
                box = [(143, 83), (181, 83), (181, 127), (143, 127)]
                closed_dashes(draw, box, phase)
                point = lerp((162, 105), (218, 105), min(1, t/.72))
            else:
                image = damaged.copy()
                clean_mask = Image.new("L", SIZE, 0)
                clean_draw = ImageDraw.Draw(clean_mask)
                limit = 1 if tool == "spot_healing" else len(spots)
                for n, (x, y, r) in enumerate(spots[:limit]):
                    local_progress = max(0.0, min(1.0, t * limit - n))
                    clean_draw.ellipse(
                        (x-r-4, y-r-4, x+r+4, y+r+4),
                        fill=int(255 * local_progress),
                    )
                image = Image.composite(base, image, clean_mask)
                point = lerp((41, 106), (98 if limit > 1 else 41, 107), t)
            cursor(image, point, ring=10)
        elif tool == "fill":
            image = base.copy()
            draw = ImageDraw.Draw(image, "RGBA")
            draw.ellipse((78, 37, 220, 139), fill=(24, 139, 203, int(185*t)), outline="#ffffff", width=3)
            cursor(image, (149, 88))
        elif tool == "gradient":
            image = base.copy()
            overlay = Image.new("RGBA", SIZE, (0, 0, 0, 0))
            pix = overlay.load()
            limit = max(1, int(288*t))
            for x in range(limit):
                blend = x / 287
                color = (int(23 + 224*blend), int(118 + 76*blend), int(220 - 164*blend), 178)
                for y in range(162):
                    pix[x, y] = color
            image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
            cursor(image, (limit-1, 81))
            ImageDraw.Draw(image).line((8, 81, limit-1, 81), fill="#ffffff", width=2)
        elif tool == "text":
            image = base.copy()
            text = "Текст"
            count = max(0, min(len(text), int(t*(len(text)+1))))
            draw = ImageDraw.Draw(image)
            draw.rectangle((65, 50, 224, 116), fill=(255, 255, 255, 205), outline="#2384d3", width=2)
            draw.text((78, 65), text[:count], font=font(32, True), fill="#17202a")
            caret_x = 78 + draw.textlength(text[:count], font=font(32, True))
            draw.line((caret_x, 66, caret_x, 103), fill="#17202a", width=2)
            cursor(image, lerp((218, 115), (caret_x + 5, 105), t))
        elif tool == "eyedropper":
            image = base.copy()
            point = lerp((168, 106), (45, 107), t)
            cursor(image, point)
            draw = ImageDraw.Draw(image)
            swatch = "#d72e25" if t > .65 else "#2bb8bc"
            draw.rectangle((238, 12, 276, 50), fill="#ffffff", outline="#111111", width=2)
            draw.rectangle((243, 17, 271, 45), fill=swatch)
        elif tool in {"rect_shape", "ellipse_shape", "line_shape", "bezier_shape", "polygon_shape", "star_shape", "custom_shape"}:
            image = landscape.copy()
            point = draw_shape(image, tool, t, phase)
            cursor(image, point)
        elif tool in {"path_select", "direct_select", "add_anchor", "delete_anchor", "convert_anchor"}:
            image = landscape.copy()
            draw = ImageDraw.Draw(image, "RGBA")
            nodes = [(55, 120), (112, 45), (186, 116), (238, 48)]
            handles = [(82, 39), (157, 142)]
            curve = []
            for step in range(61):
                u = step / 60
                x = (1-u)**3*nodes[0][0] + 3*(1-u)**2*u*handles[0][0] + 3*(1-u)*u*u*handles[1][0] + u**3*nodes[-1][0]
                y = (1-u)**3*nodes[0][1] + 3*(1-u)**2*u*handles[0][1] + 3*(1-u)*u*u*handles[1][1] + u**3*nodes[-1][1]
                curve.append((x, y))
            draw.line(curve, fill="#ffffff", width=4)
            active = (112, int(45 + 35*t)) if tool == "direct_select" else nodes[1]
            shown = [nodes[0], active, nodes[2], nodes[3]]
            if tool == "add_anchor" and t > .55:
                shown.insert(2, (150, 81))
            if tool == "delete_anchor" and t > .55:
                shown.pop(1)
            draw.line((shown[0], handles[0]), fill="#f0b84f", width=2)
            draw.line((handles[1], shown[-1]), fill="#f0b84f", width=2)
            for n, (x, y) in enumerate(shown):
                fill = "#2384d3" if n == min(1, len(shown) - 1) else "#ffffff"
                draw.rectangle((x-5, y-5, x+5, y+5), fill=fill, outline="#14181d", width=2)
            if tool == "convert_anchor" and t > .55:
                draw.line((active[0]-30, active[1], active[0]+30, active[1]), fill="#f0b84f", width=2)
                draw.ellipse((active[0]-34, active[1]-4, active[0]-26, active[1]+4), fill="#ffffff")
                draw.ellipse((active[0]+26, active[1]-4, active[0]+34, active[1]+4), fill="#ffffff")
            pointer = lerp((262, 142), active, t)
            draw.ellipse(
                (active[0] - 7 - int(2 * t), active[1] - 7 - int(2 * t), active[0] + 7 + int(2 * t), active[1] + 7 + int(2 * t)),
                outline=(35, 132, 211, 90 + int(150 * t)),
                width=2,
            )
            cursor(image, pointer)
        elif tool in {"select", "ellipse_select", "lasso", "magnetic_lasso", "polygon_lasso", "quick_selection", "magic_wand", "color_range"}:
            image, _ = selection_demo(base, tool, t, phase)
        else:  # crop
            image = landscape.copy()
            margin_x, margin_y = int(8 + 42*t), int(6 + 22*t)
            draw = ImageDraw.Draw(image, "RGBA")
            draw.rectangle((0, 0, 288, margin_y), fill=(0, 0, 0, 130))
            draw.rectangle((0, 162-margin_y, 288, 162), fill=(0, 0, 0, 130))
            draw.rectangle((0, margin_y, margin_x, 162-margin_y), fill=(0, 0, 0, 130))
            draw.rectangle((288-margin_x, margin_y, 288, 162-margin_y), fill=(0, 0, 0, 130))
            draw.rectangle((margin_x, margin_y, 288-margin_x, 162-margin_y), outline="#ffffff", width=3)
            for x in (margin_x, 288-margin_x):
                for y in (margin_y, 162-margin_y):
                    draw.rectangle((x-4, y-4, x+4, y+4), fill="#ffffff")
            cursor(image, (margin_x, margin_y))
        stage_banner(image, tool, index)
        frames.append(image.convert("P", palette=Image.Palette.ADAPTIVE, colors=128))
    return frames


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    still = cover(SOURCE_DIR / "still_life.png")
    landscape = cover(SOURCE_DIR / "landscape.png")
    for tool in TOOLS:
        frames = make_frames(tool, still, landscape)
        output = ASSET_DIR / f"{tool}.gif"
        temporary = ASSET_DIR / f".{tool}.tmp.gif"
        previous = ASSET_DIR / f".{tool}.previous.gif"
        frames[0].save(
            temporary,
            save_all=True,
            append_images=frames[1:],
            duration=110,
            loop=0,
            optimize=True,
            disposal=2,
        )
        previous.unlink(missing_ok=True)
        if output.exists():
            output.replace(previous)
        os.replace(temporary, output)
        previous.unlink(missing_ok=True)
        print(f"created {tool}.gif")


if __name__ == "__main__":
    main()
