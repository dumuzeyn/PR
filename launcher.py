def enable_windows_hidpi() -> None:
    try:
        import ctypes

        awareness_context_per_monitor_v2 = ctypes.c_void_p(-4)
        if ctypes.windll.user32.SetProcessDpiAwarenessContext(awareness_context_per_monitor_v2):
            return
    except Exception:
        pass
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            import ctypes

            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


enable_windows_hidpi()

from photoredactor.app import main


if __name__ == "__main__":
    main()
