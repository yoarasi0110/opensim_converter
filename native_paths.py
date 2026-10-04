"""Keep OpenSim's Windows narrow filename API away from Unicode paths."""
from pathlib import Path
import ctypes
import shutil
import tempfile

_cache = {}
_directories = []  # TemporaryDirectory cleans its own directory at interpreter exit.


def short_path(path):
    buffer = ctypes.create_unicode_buffer(32768)
    if ctypes.windll.kernel32.GetShortPathNameW(str(path), buffer, len(buffer)):
        result = Path(buffer.value)
        if str(result).isascii():
            return result
    return None


def model_path(path):
    path = Path(path).resolve()
    if str(path).isascii():
        return path
    if path in _cache:
        return _cache[path]
    short = short_path(path)
    if short:
        _cache[path] = short
        return short
    base = Path(tempfile.gettempdir())
    if not str(base).isascii():
        base = short_path(base)
        if base is None:
            raise ValueError("OpenSim 需要英文暫存路徑；請設定 TEMP 為可寫入的英文資料夾後再啟動")
    directory = tempfile.TemporaryDirectory(prefix="OpenSimConverter_", dir=base,
                                          ignore_cleanup_errors=True)
    _directories.append(directory)
    destination = Path(directory.name) / "model.osim"
    shutil.copy2(path, destination)
    _cache[path] = destination
    return destination
