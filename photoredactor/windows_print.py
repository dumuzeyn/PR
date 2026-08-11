from __future__ import annotations

from dataclasses import dataclass
import ctypes
from ctypes import wintypes
import os

import numpy as np


PD_NOSELECTION = 0x00000004
PD_NOPAGENUMS = 0x00000008
PD_RETURNDC = 0x00000100
PD_USEDEVMODECOPIESANDCOLLATE = 0x00040000
PD_DISABLEPRINTTOFILE = 0x00080000
HORZRES, VERTRES = 8, 10
LOGPIXELSX, LOGPIXELSY = 88, 90
PHYSICALWIDTH, PHYSICALHEIGHT = 110, 111
PHYSICALOFFSETX, PHYSICALOFFSETY = 112, 113
DIB_RGB_COLORS, BI_RGB, SRCCOPY, HALFTONE = 0, 0, 0x00CC0020, 4


class PRINTDLGW(ctypes.Structure):
    _fields_ = [
        ("lStructSize", wintypes.DWORD),
        ("hwndOwner", wintypes.HWND),
        ("hDevMode", wintypes.HGLOBAL),
        ("hDevNames", wintypes.HGLOBAL),
        ("hDC", wintypes.HDC),
        ("Flags", wintypes.DWORD),
        ("nFromPage", wintypes.WORD),
        ("nToPage", wintypes.WORD),
        ("nMinPage", wintypes.WORD),
        ("nMaxPage", wintypes.WORD),
        ("nCopies", wintypes.WORD),
        ("hInstance", wintypes.HINSTANCE),
        ("lCustData", wintypes.LPARAM),
        ("lpfnPrintHook", ctypes.c_void_p),
        ("lpfnSetupHook", ctypes.c_void_p),
        ("lpPrintTemplateName", wintypes.LPCWSTR),
        ("lpSetupTemplateName", wintypes.LPCWSTR),
        ("hPrintTemplate", wintypes.HGLOBAL),
        ("hSetupTemplate", wintypes.HGLOBAL),
    ]


class DOCINFOW(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_int),
        ("lpszDocName", wintypes.LPCWSTR),
        ("lpszOutput", wintypes.LPCWSTR),
        ("lpszDatatype", wintypes.LPCWSTR),
        ("fwType", wintypes.DWORD),
    ]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", ctypes.c_long),
        ("biHeight", ctypes.c_long),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", ctypes.c_long),
        ("biYPelsPerMeter", ctypes.c_long),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]


@dataclass(frozen=True)
class PrinterPage:
    printable_width: int
    printable_height: int
    offset_x: int
    offset_y: int
    dpi_x: int
    dpi_y: int


@dataclass(frozen=True)
class PrintPlacement:
    x: int
    y: int
    width: int
    height: int


def calculate_placement(
    image_width: int,
    image_height: int,
    document_dpi: float,
    page: PrinterPage,
    fit_to_page: bool = True,
) -> PrintPlacement:
    if min(image_width, image_height, page.printable_width, page.printable_height) <= 0:
        raise ValueError("Размер изображения и печатной области должен быть положительным")
    if fit_to_page:
        scale = min(page.printable_width / image_width, page.printable_height / image_height)
    else:
        if document_dpi <= 0:
            raise ValueError("DPI документа должен быть положительным")
        scale = min(page.dpi_x, page.dpi_y) / float(document_dpi)
    width = max(1, round(image_width * scale))
    height = max(1, round(image_height * scale))
    x = page.offset_x + (page.printable_width - width) // 2
    y = page.offset_y + (page.printable_height - height) // 2
    return PrintPlacement(x, y, width, height)


def rgba_to_bgra_on_paper(pixels: np.ndarray, paper: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    source = np.asarray(pixels)
    if source.ndim != 3 or source.shape[2] != 4:
        raise ValueError("Для печати требуется RGBA-изображение")
    alpha = source[:, :, 3:4].astype(np.float32) / 255.0
    rgb = np.clip(source[:, :, :3].astype(np.float32) * alpha + np.array(paper, dtype=np.float32) * (1.0 - alpha), 0, 255)
    result = np.empty(source.shape, dtype=np.uint8)
    result[:, :, :3] = rgb[:, :, ::-1].astype(np.uint8)
    result[:, :, 3] = 255
    return np.ascontiguousarray(result)


class WindowsPrinter:
    def __init__(self) -> None:
        if os.name != "nt":
            raise RuntimeError("Системная физическая печать доступна только в Windows")
        self.comdlg32 = ctypes.WinDLL("comdlg32", use_last_error=True)
        self.gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._declare_signatures()

    def _declare_signatures(self) -> None:
        self.comdlg32.PrintDlgW.argtypes = [ctypes.POINTER(PRINTDLGW)]
        self.comdlg32.PrintDlgW.restype = wintypes.BOOL
        self.comdlg32.CommDlgExtendedError.restype = wintypes.DWORD
        self.gdi32.GetDeviceCaps.argtypes = [wintypes.HDC, ctypes.c_int]
        self.gdi32.GetDeviceCaps.restype = ctypes.c_int
        self.gdi32.StartDocW.argtypes = [wintypes.HDC, ctypes.POINTER(DOCINFOW)]
        self.gdi32.StartDocW.restype = ctypes.c_int
        for name in ("StartPage", "EndPage", "EndDoc", "AbortDoc", "DeleteDC"):
            function = getattr(self.gdi32, name)
            function.argtypes = [wintypes.HDC]
            function.restype = ctypes.c_int
        self.gdi32.SetStretchBltMode.argtypes = [wintypes.HDC, ctypes.c_int]
        self.gdi32.StretchDIBits.argtypes = [
            wintypes.HDC,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.POINTER(BITMAPINFO),
            wintypes.UINT,
            wintypes.DWORD,
        ]
        self.gdi32.StretchDIBits.restype = ctypes.c_int
        self.kernel32.GlobalFree.argtypes = [wintypes.HGLOBAL]
        self.kernel32.GlobalFree.restype = wintypes.HGLOBAL

    def choose_printer(self, owner: int = 0) -> PRINTDLGW | None:
        dialog = PRINTDLGW()
        dialog.lStructSize = ctypes.sizeof(PRINTDLGW)
        dialog.hwndOwner = owner
        dialog.Flags = PD_RETURNDC | PD_USEDEVMODECOPIESANDCOLLATE | PD_NOPAGENUMS | PD_NOSELECTION | PD_DISABLEPRINTTOFILE
        dialog.nMinPage = dialog.nMaxPage = dialog.nFromPage = dialog.nToPage = 1
        if self.comdlg32.PrintDlgW(ctypes.byref(dialog)):
            return dialog
        error = int(self.comdlg32.CommDlgExtendedError())
        if error:
            raise OSError(error, "Windows не смог открыть системный диалог печати")
        return None

    def page_details(self, hdc: int) -> PrinterPage:
        get = lambda key: int(self.gdi32.GetDeviceCaps(hdc, key))
        return PrinterPage(get(HORZRES), get(VERTRES), get(PHYSICALOFFSETX), get(PHYSICALOFFSETY), get(LOGPIXELSX), get(LOGPIXELSY))

    def print_rgba(
        self,
        pixels: np.ndarray,
        document_name: str,
        document_dpi: float,
        owner: int = 0,
        fit_to_page: bool = True,
    ) -> bool:
        dialog = self.choose_printer(owner)
        if dialog is None:
            return False
        hdc = dialog.hDC
        started = False
        try:
            page = self.page_details(hdc)
            placement = calculate_placement(pixels.shape[1], pixels.shape[0], document_dpi, page, fit_to_page)
            bitmap = rgba_to_bgra_on_paper(pixels)
            height, width = bitmap.shape[:2]
            header = BITMAPINFOHEADER(
                ctypes.sizeof(BITMAPINFOHEADER), width, -height, 1, 32, BI_RGB, bitmap.nbytes, 0, 0, 0, 0
            )
            info = BITMAPINFO(header, (wintypes.DWORD * 3)(0, 0, 0))
            doc_info = DOCINFOW(ctypes.sizeof(DOCINFOW), document_name, None, None, 0)
            if self.gdi32.StartDocW(hdc, ctypes.byref(doc_info)) <= 0:
                raise ctypes.WinError(ctypes.get_last_error())
            started = True
            if self.gdi32.StartPage(hdc) <= 0:
                raise ctypes.WinError(ctypes.get_last_error())
            self.gdi32.SetStretchBltMode(hdc, HALFTONE)
            result = self.gdi32.StretchDIBits(
                hdc,
                placement.x,
                placement.y,
                placement.width,
                placement.height,
                0,
                0,
                width,
                height,
                bitmap.ctypes.data_as(ctypes.c_void_p),
                ctypes.byref(info),
                DIB_RGB_COLORS,
                SRCCOPY,
            )
            if result in {0, -1}:
                raise ctypes.WinError(ctypes.get_last_error())
            if self.gdi32.EndPage(hdc) <= 0 or self.gdi32.EndDoc(hdc) <= 0:
                raise ctypes.WinError(ctypes.get_last_error())
            started = False
            return True
        except Exception:
            if started:
                self.gdi32.AbortDoc(hdc)
            raise
        finally:
            if hdc:
                self.gdi32.DeleteDC(hdc)
            if dialog.hDevMode:
                self.kernel32.GlobalFree(dialog.hDevMode)
            if dialog.hDevNames:
                self.kernel32.GlobalFree(dialog.hDevNames)


def print_document(document, owner: int = 0, fit_to_page: bool = True) -> bool:
    name = "PhotoRedactor"
    if document.path:
        from pathlib import Path

        name = Path(document.path).stem
    return WindowsPrinter().print_rgba(document.composite(False), name, document.dpi, owner, fit_to_page)


__all__ = [name for name in globals() if not name.startswith("__")]
