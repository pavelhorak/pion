"""The data directory every persistence file is named under (embedded Pion).

`data_path("pion.wal.0")` is `<dir>/pion.wal.0` once a directory is set
(`pion_set_data_dir`, fcntl_wrap.c), and `pion.wal.0` otherwise: the server
sets none, so its files stay where they have always been, relative to the
working directory. Cold: called when a file is opened, never per command."""
from std.ffi import external_call
from src.common.ptr import is_null


def data_path(name: String) -> String:
    var n = Int(external_call["pion_data_dir_len", Int64]())
    if n == 0:
        return name
    var p = external_call["pion_data_dir_ptr", Pointer[UInt8, MutUntrackedOrigin]]()
    if is_null(p):
        return name
    var prefix = String(StringSpan[MutUntrackedOrigin](
        unsafe_from_utf8=Span[UInt8, MutUntrackedOrigin](unsafe_ptr=p, length=n)))
    return prefix + name
