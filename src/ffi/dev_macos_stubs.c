/* dev_macos_stubs.c — stubs for Linux-only symbols (epoll, glibc errno).
 *
 * The default `pixi run build` runs Mojo at -O3, where dead-code
 * elimination removes references to Linux-only paths (run_server_epoll,
 * run_server_xdp) on macOS targets. The dev path `pixi run build-dev`
 * runs at -O0 (4-5× faster compile) where DCE doesn't fire — this file
 * provides do-nothing stubs so the link succeeds.
 *
 * These stubs MUST NOT be reached at runtime on macOS — the dispatch
 * checks (`if config.network_engine == "epoll"` etc.) keep the Mac
 * default tier (kqueue) alive. The dev binary is for iteration only:
 * "does it compile, do basic tests pass" — never benchmark a -O0 build.
 */

#include <errno.h>
#include <stdint.h>
#include <stddef.h>

/* Linux glibc internal accessor for thread-local `errno`. macOS uses
 * `__error()` instead; provide a shim that returns the address of the
 * thread's errno so any caller that takes this symbol's signature
 * (returning int*) gets a valid (if unused) location. */
int *__errno_location(void) {
    return &errno;
}

/* Linux epoll syscalls. Return -1 + EPERM so any accidental call surfaces
 * loudly rather than silently succeeding. None of these should fire on
 * macOS — the engine selects kqueue at startup. */
int epoll_create1(int flags) {
    (void)flags;
    errno = EPERM;
    return -1;
}

int epoll_ctl(int epfd, int op, int fd, void *event) {
    (void)epfd; (void)op; (void)fd; (void)event;
    errno = EPERM;
    return -1;
}

int epoll_wait(int epfd, void *events, int maxevents, int timeout) {
    (void)epfd; (void)events; (void)maxevents; (void)timeout;
    errno = EPERM;
    return -1;
}
