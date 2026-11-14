/* _pion: the CPython binding to libpion (stable ABI, CPython >= 3.11).
 *
 * The ctypes binding (_engine.py) works; this one removes its per-call cost.
 * libpion is dlopen'ed RTLD_GLOBAL (lua_wrap.c finds pion_script_dispatch
 * with dlsym(RTLD_DEFAULT)), its C ABI is resolved once, and every engine call
 * runs with the GIL released: the engine has its own mutex.
 */
#define PY_SSIZE_T_CLEAN
#define Py_LIMITED_API 0x030B0000
#include <Python.h>
#include <dlfcn.h>
#include <stdint.h>

typedef int64_t (*open_fn)(const char *, int64_t, char *, int64_t);
typedef int64_t (*conn_new_fn)(int64_t);
typedef void (*conn_free_fn)(int64_t, int64_t);
typedef int64_t (*exec_fn)(int64_t, int64_t, const char *, int64_t, const char **, int64_t *);
typedef int64_t (*poll_fn)(int64_t, int64_t, const char **, int64_t *);
typedef int64_t (*h_fn)(int64_t);
typedef int64_t (*version_fn)(char *, int64_t);

static open_fn f_open;
static conn_new_fn f_conn_new;
static conn_free_fn f_conn_free;
static exec_fn f_exec;
static poll_fn f_poll;
static h_fn f_tick, f_close;
static version_fn f_version;

static PyObject *load(PyObject *self, PyObject *args) {
    const char *path;
    if (!PyArg_ParseTuple(args, "s", &path)) return NULL;
    void *h = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
    if (!h) { PyErr_Format(PyExc_OSError, "dlopen %s: %s", path, dlerror()); return NULL; }
#define SYM(var, type, name) var = (type)dlsym(h, name); \
    if (!var) { PyErr_Format(PyExc_OSError, "libpion lacks %s", name); return NULL; }
    SYM(f_open, open_fn, "pion_embed_open")
    SYM(f_conn_new, conn_new_fn, "pion_embed_conn_new")
    SYM(f_conn_free, conn_free_fn, "pion_embed_conn_free")
    SYM(f_exec, exec_fn, "pion_embed_exec")
    SYM(f_poll, poll_fn, "pion_embed_poll")
    SYM(f_tick, h_fn, "pion_embed_tick")
    SYM(f_close, h_fn, "pion_embed_close")
    SYM(f_version, version_fn, "pion_embed_version")
#undef SYM
    Py_RETURN_NONE;
}

static PyObject *py_open(PyObject *self, PyObject *args) {
    /* y* (a buffer), never y#: an extension built against 3.13+ headers
     * calls PyArg_ParseTuple without the size_t remap, and 3.11 then refuses
     * every '#' format (PY_SSIZE_T_CLEAN). */
    Py_buffer cfg;
    if (!PyArg_ParseTuple(args, "y*", &cfg)) return NULL;
    char err[1024]; err[0] = 0;
    int64_t h;
    Py_BEGIN_ALLOW_THREADS
    h = f_open((const char *)cfg.buf, (int64_t)cfg.len, err, sizeof err);
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&cfg);
    return Py_BuildValue("(Ls)", (long long)h, err);
}

static PyObject *py_conn_new(PyObject *self, PyObject *args) {
    long long h; if (!PyArg_ParseTuple(args, "L", &h)) return NULL;
    int64_t c;
    Py_BEGIN_ALLOW_THREADS
    c = f_conn_new((int64_t)h);
    Py_END_ALLOW_THREADS
    return PyLong_FromLongLong((long long)c);
}

static PyObject *py_conn_free(PyObject *self, PyObject *args) {
    long long h, c; if (!PyArg_ParseTuple(args, "LL", &h, &c)) return NULL;
    Py_BEGIN_ALLOW_THREADS
    f_conn_free((int64_t)h, (int64_t)c);
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

/* The reply bytes stay valid until the next call on this connection; they
 * are copied into a bytes object before anything else can run on it. */
static PyObject *reply(int64_t st, const char *out, int64_t n) {
    PyObject *b = PyBytes_FromStringAndSize(n > 0 ? out : "", n > 0 ? (Py_ssize_t)n : 0);
    if (!b) return NULL;
    PyObject *r = Py_BuildValue("(LN)", (long long)st, b);
    return r;
}

static PyObject *py_exec(PyObject *self, PyObject *args) {
    long long h, c; Py_buffer buf;
    if (!PyArg_ParseTuple(args, "LLy*", &h, &c, &buf)) return NULL;
    const char *out = NULL; int64_t n = 0, st;
    Py_BEGIN_ALLOW_THREADS
    st = f_exec((int64_t)h, (int64_t)c, (const char *)buf.buf, (int64_t)buf.len, &out, &n);
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&buf);
    return reply(st, out, n);
}

static PyObject *py_poll(PyObject *self, PyObject *args) {
    long long h, c; if (!PyArg_ParseTuple(args, "LL", &h, &c)) return NULL;
    const char *out = NULL; int64_t n = 0, st;
    Py_BEGIN_ALLOW_THREADS
    st = f_poll((int64_t)h, (int64_t)c, &out, &n);
    Py_END_ALLOW_THREADS
    return reply(st, out, n);
}

static PyObject *py_tick(PyObject *self, PyObject *args) {
    long long h; if (!PyArg_ParseTuple(args, "L", &h)) return NULL;
    int64_t r;
    Py_BEGIN_ALLOW_THREADS
    r = f_tick((int64_t)h);
    Py_END_ALLOW_THREADS
    return PyLong_FromLongLong((long long)r);
}

static PyObject *py_close(PyObject *self, PyObject *args) {
    long long h; if (!PyArg_ParseTuple(args, "L", &h)) return NULL;
    int64_t r;
    Py_BEGIN_ALLOW_THREADS
    r = f_close((int64_t)h);
    Py_END_ALLOW_THREADS
    return PyLong_FromLongLong((long long)r);
}

static PyObject *py_version(PyObject *self, PyObject *args) {
    char b[128]; b[0] = 0;
    f_version(b, sizeof b);
    return PyUnicode_FromString(b);
}

static PyMethodDef methods[] = {
    {"load", load, METH_VARARGS, "dlopen libpion (RTLD_GLOBAL) and resolve its C ABI"},
    {"open", py_open, METH_VARARGS, "open(cfg: bytes) -> (handle, error)"},
    {"conn_new", py_conn_new, METH_VARARGS, NULL},
    {"conn_free", py_conn_free, METH_VARARGS, NULL},
    {"exec", py_exec, METH_VARARGS, "exec(h, conn, request) -> (status, reply)"},
    {"poll", py_poll, METH_VARARGS, "poll(h, conn) -> (status, reply)"},
    {"tick", py_tick, METH_VARARGS, NULL},
    {"close", py_close, METH_VARARGS, NULL},
    {"version", py_version, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef moddef = {PyModuleDef_HEAD_INIT, "_pion", NULL, -1, methods};

PyMODINIT_FUNC PyInit__pion(void) { return PyModule_Create(&moddef); }
