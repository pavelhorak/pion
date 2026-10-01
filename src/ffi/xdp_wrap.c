/* xdp_wrap.c — AF_XDP socket setup and ring management for Pion
 *
 * Provides C functions called from Mojo for:
 *   1. Loading the XDP BPF program onto a NIC
 *   2. Creating AF_XDP sockets with UMEM (shared memory for zero-copy I/O)
 *   3. Polling RX/TX rings for packet send/receive
 *   4. TCP-Lite packet construction (SYN-ACK, ACK, data responses)
 *
 * Architecture:
 *   Mojo event loop → pion_xdp_poll_rx() → raw Ethernet frames
 *                   → extract TCP payload → fast_path.process_data_plane()
 *                   → pion_xdp_send_frame() → TX ring → NIC
 *
 * Requirements: Linux 5.4+ with AF_XDP support.
 * Build: gcc -O2 -c src/ffi/xdp_wrap.c -o src/ffi/xdp_wrap.o
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>

#ifdef __linux__

#include <sys/socket.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <linux/if_link.h>
#include <linux/if_xdp.h>
#include <linux/bpf.h>
#include <net/if.h>
#include <poll.h>
#include <arpa/inet.h>
#include <linux/netlink.h>
#include <linux/rtnetlink.h>
#include <sys/syscall.h>
#include <signal.h>

/* Forward declarations for signal handler */
void pion_xdp_detach_by_name(const char *ifname);
void pion_xdp_remove_flow_steering(const char *ifname, uint16_t target_port);

/* Forward declaration for pion_xdp_send_fragmented (uses this) */
int pion_xdp_build_tcp_response(void *frame_buf, const void *src_frame,
                                 int src_frame_len,
                                 uint8_t tcp_flags,
                                 uint32_t seq_num, uint32_t ack_num,
                                 const void *payload, int payload_len);

/* ── Signal handler state for BPF cleanup ─────────────────────────────────── */

static volatile int g_xdp_ifindex = 0;
static char g_xdp_ifname[64] = {0};
static uint16_t g_xdp_port = 0;

static void _xdp_signal_handler(int sig) {
    /* Detach BPF program from NIC on SIGINT/SIGTERM */
    if (g_xdp_ifindex > 0) {
        pion_xdp_detach_by_name(g_xdp_ifname);
        if (g_xdp_port > 0)
            pion_xdp_remove_flow_steering(g_xdp_ifname, g_xdp_port);
        g_xdp_ifindex = 0;
    }
    /* Re-raise to get default behavior (exit) */
    signal(sig, SIG_DFL);
    raise(sig);
}

/* Register signal handlers for graceful XDP cleanup.
 * Called from Mojo after BPF attach succeeds. */
void pion_xdp_register_signal_handlers(const char *ifname, uint16_t target_port) {
    if (!ifname) return;
    strncpy(g_xdp_ifname, ifname, sizeof(g_xdp_ifname) - 1);
    g_xdp_ifindex = (int)if_nametoindex(ifname);
    g_xdp_port = target_port;

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = _xdp_signal_handler;
    sa.sa_flags = 0;  /* no SA_RESTART — we want to exit */
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    printf("XDP: signal handlers registered for %s (SIGINT/SIGTERM → BPF detach)\n", ifname);
}

/* ── BPF syscall wrapper ──────────────────────────────────────────────────── */

static inline int bpf_syscall(int cmd, union bpf_attr *attr, unsigned int size) {
    return (int)syscall(__NR_bpf, cmd, attr, size);
}

/* ── UMEM and Ring Structures ─────────────────────────────────────────────── */

#define PION_XDP_NUM_FRAMES    4096
#define PION_XDP_FRAME_SIZE    4096
#define PION_XDP_UMEM_SIZE     (PION_XDP_NUM_FRAMES * PION_XDP_FRAME_SIZE)  /* 16MB */
#define PION_XDP_RING_SIZE     2048
#define PION_XDP_BATCH_SIZE    64

/* AF_XDP ring producer/consumer offsets (from getsockopt XDP_MMAP_OFFSETS) */
struct pion_xdp_ring_offsets {
    uint64_t producer;
    uint64_t consumer;
    uint64_t desc;
    uint64_t flags;
};

/* Main XDP context — one per worker */
typedef struct {
    int xsk_fd;                              /* AF_XDP socket fd */
    int ifindex;                             /* NIC interface index */
    int queue_id;                            /* RX queue bound to this socket */
    uint16_t target_port;                    /* Pion port (host byte order) */

    /* UMEM: contiguous memory region for zero-copy frame I/O */
    void *umem_area;                         /* mmap'd UMEM base */
    uint64_t umem_size;

    /* Fill ring: userspace → kernel (provides empty frames for RX) */
    uint32_t *fill_producer;
    uint32_t *fill_consumer;
    uint64_t *fill_ring;                     /* array of frame addresses */
    uint32_t  fill_mask;

    /* Completion ring: kernel → userspace (frames sent by TX, can be reused) */
    uint32_t *comp_producer;
    uint32_t *comp_consumer;
    uint64_t *comp_ring;
    uint32_t  comp_mask;

    /* RX ring: kernel → userspace (received frames) */
    uint32_t *rx_producer;
    uint32_t *rx_consumer;
    struct xdp_desc *rx_ring;
    uint32_t  rx_mask;

    /* TX ring: userspace → kernel (frames to transmit) */
    uint32_t *tx_producer;
    uint32_t *tx_consumer;
    struct xdp_desc *tx_ring;
    uint32_t  tx_mask;

    /* Free frame stack for UMEM frame management */
    uint64_t  free_frames[PION_XDP_NUM_FRAMES];
    uint32_t  free_frame_count;

    /* BPF program fd and XSKMAP fd (for cleanup) */
    int bpf_prog_fd;
    int xskmap_fd;

    /* Stats */
    uint64_t rx_packets;
    uint64_t tx_packets;
    uint64_t rx_bytes;
    uint64_t tx_bytes;
} PionXDPContext;

/* ── Frame management ─────────────────────────────────────────────────────── */

static uint64_t _xdp_alloc_frame(PionXDPContext *ctx) {
    if (ctx->free_frame_count == 0) return UINT64_MAX;
    return ctx->free_frames[--ctx->free_frame_count];
}

static void _xdp_free_frame(PionXDPContext *ctx, uint64_t addr) {
    if (ctx->free_frame_count < PION_XDP_NUM_FRAMES)
        ctx->free_frames[ctx->free_frame_count++] = addr;
}

/* ── Fill ring: provide empty frames to kernel for RX ─────────────────────── */

static void _xdp_populate_fill_ring(PionXDPContext *ctx, uint32_t count) {
    uint32_t idx = *ctx->fill_producer;
    for (uint32_t i = 0; i < count; i++) {
        uint64_t addr = _xdp_alloc_frame(ctx);
        if (addr == UINT64_MAX) break;
        ctx->fill_ring[(idx + i) & ctx->fill_mask] = addr;
    }
    __sync_synchronize();  /* memory barrier before updating producer */
    *ctx->fill_producer = idx + count;
}

/* ── BPF program loading (minimal loader, no libbpf dependency) ────────── */

static int _xdp_load_bpf_prog(uint16_t target_port_be, int xskmap_fd) {
    /* Minimal XDP program in BPF bytecode that redirects TCP packets on the target
     * port to an XSKMAP. The XSKMAP fd is passed in by the caller (who also
     * populates it with the AF_XDP socket fd after bind()).
     *
     * Register usage: r6=ctx, r2=data, r3=data_end, r1=scratch
     */

    /* BPF instructions for the XDP filter program.
     * This is the compiled form of the logic in xdp_kern.c.
     * Port value is patched in at instruction index 21 (IMM field). */
    struct bpf_insn {
        uint8_t  code;
        uint8_t  dst_src;   /* dst:4, src:4 */
        int16_t  off;
        int32_t  imm;
    };

    /* XDP BPF program: redirect TCP packets on target port to AF_XDP socket.
     * All other packets pass through to the kernel stack.
     *
     * The BPF verifier tracks packet bounds — each packet read must be preceded
     * by a bounds check that proves the offset is within [data, data_end).
     * We use a single conservative check: data + 54 <= data_end (eth 14 + IP 20 + TCP 20).
     * This covers all fixed-offset reads. Variable IHL is handled with a second check.
     *
     * Jump offsets: BPF "goto +N" = skip N insns from NEXT insn.
     * pass@25, redirect@19. 27 instructions total. */
    struct bpf_insn prog[] = {
        /* r6 = ctx (save for later) */
        { 0xbf, 0x16, 0, 0 },              /* 0:  r6 = r1 */
        /* r2 = data, r3 = data_end */
        { 0x61, 0x12, 0, 0 },              /* 1:  r2 = *(u32 *)(r1 + 0) */
        { 0x61, 0x13, 4, 0 },              /* 2:  r3 = *(u32 *)(r1 + 4) */
        /* Bounds check: data + 54 <= data_end (covers eth + min IP + min TCP) */
        { 0xbf, 0x21, 0, 0 },              /* 3:  r1 = r2 */
        { 0x07, 0x01, 0, 54 },             /* 4:  r1 += 54 */
        { 0x2d, 0x31, 14, 0 },             /* 5:  if r1 > r3 goto pass@20 */
        /* Check eth_proto == IPv4 (0x0008 in network byte order) */
        { 0x69, 0x24, 12, 0 },             /* 6:  r4 = *(u16 *)(r2 + 12) */
        { 0x55, 0x04, 12, 0x0008 },        /* 7:  if r4 != 0x0008 goto pass@20 */
        /* Check IP protocol == TCP (6) at eth(14) + 9 = offset 23 */
        { 0x71, 0x24, 23, 0 },             /* 8:  r4 = *(u8 *)(r2 + 23) */
        { 0x55, 0x04, 10, 6 },             /* 9:  if r4 != 6 goto pass@20 */
        /* TCP dest port at fixed offset 36 (eth 14 + IP 20 + TCP offset 2)
         * Use r2 (data ptr), NOT r1 (which is data+54 after bounds check).
         * Assumes IHL=5 (20 bytes). IP options → wrong port → XDP_PASS (safe). */
        { 0x69, 0x24, 36, 0 },             /* 10: r4 = *(u16 *)(r2 + 36) [tcp->dest] */
        { 0x69, 0x25, 34, 0 },             /* 11: r5 = *(u16 *)(r2 + 34) [tcp->source] */
        /* if dest_port == target goto redirect@14 (ldimm64 start) */
        { 0x15, 0x04, 1, target_port_be }, /* 12: if r4 == port goto +1 → insn 14 */
        /* if src_port != target goto pass@20 */
        { 0x55, 0x05, 6, target_port_be }, /* 13: if r5 != port goto +6 → insn 20 */
        /* fall through to redirect (source port matched) */

        /* redirect@14: r1 = xskmap_fd (BPF_LD_MAP_FD, 2 insns) */
        { 0x18, 0x11, 0, xskmap_fd },      /* 14: r1 = map_fd (lo) */
        { 0x00, 0x00, 0, 0 },              /* 15: (hi, imm64 continuation) */
        { 0x61, 0x62, 16, 0 },             /* 16: r2 = *(u32 *)(r6 + 16) [rx_queue_index] */
        { 0xb7, 0x03, 0, 0 },              /* 17: r3 = 0 */
        { 0x85, 0x00, 0, 51 },             /* 18: call bpf_redirect_map */
        { 0x95, 0x00, 0, 0 },              /* 19: exit */

        /* pass@20: return XDP_PASS (2) */
        { 0xb7, 0x00, 0, 2 },              /* 20: r0 = 2 */
        { 0x95, 0x00, 0, 0 },              /* 21: exit */
    };

    /* Load the BPF program */
    char log_buf[4096];
    memset(log_buf, 0, sizeof(log_buf));
    union bpf_attr prog_attr;
    memset(&prog_attr, 0, sizeof(prog_attr));
    prog_attr.prog_type  = BPF_PROG_TYPE_XDP;
    prog_attr.insn_cnt   = sizeof(prog) / sizeof(prog[0]);
    prog_attr.insns      = (uint64_t)(unsigned long)prog;
    prog_attr.license    = (uint64_t)(unsigned long)"Dual BSD/GPL";
    prog_attr.log_level  = 1;
    prog_attr.log_size   = sizeof(log_buf);
    prog_attr.log_buf    = (uint64_t)(unsigned long)log_buf;

    int prog_fd = bpf_syscall(BPF_PROG_LOAD, &prog_attr, sizeof(prog_attr));
    if (prog_fd < 0) {
        fprintf(stderr, "XDP: BPF program load failed (errno=%d): %s\n", errno, log_buf);
        return -1;
    }

    return prog_fd;
}

/* ── Public API ───────────────────────────────────────────────────────────── */

void *pion_xdp_create(const char *ifname, int queue_id, uint16_t target_port) {
    PionXDPContext *ctx = (PionXDPContext *)calloc(1, sizeof(PionXDPContext));
    if (!ctx) return NULL;

    ctx->ifindex = (int)if_nametoindex(ifname);
    if (ctx->ifindex == 0) {
        fprintf(stderr, "XDP: interface '%s' not found\n", ifname);
        free(ctx);
        return NULL;
    }
    ctx->queue_id = queue_id;
    ctx->target_port = target_port;
    ctx->xsk_fd = -1;
    ctx->bpf_prog_fd = -1;
    ctx->xskmap_fd = -1;

    /* Allocate UMEM area — 16MB page-aligned region for zero-copy frame I/O */
    ctx->umem_area = mmap(NULL, PION_XDP_UMEM_SIZE, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
    if (ctx->umem_area == MAP_FAILED) {
        fprintf(stderr, "XDP: UMEM mmap failed (size=%d)\n", PION_XDP_UMEM_SIZE);
        free(ctx);
        return NULL;
    }
    ctx->umem_size = PION_XDP_UMEM_SIZE;

    /* Initialize free frame stack (all frames available) */
    ctx->free_frame_count = PION_XDP_NUM_FRAMES;
    for (uint32_t i = 0; i < PION_XDP_NUM_FRAMES; i++)
        ctx->free_frames[i] = i * PION_XDP_FRAME_SIZE;

    /* Create AF_XDP socket */
    ctx->xsk_fd = socket(AF_XDP, SOCK_RAW, 0);
    if (ctx->xsk_fd < 0) {
        fprintf(stderr, "XDP: AF_XDP socket creation failed (errno=%d). Requires Linux 5.4+ and CAP_NET_ADMIN.\n", errno);
        munmap(ctx->umem_area, PION_XDP_UMEM_SIZE);
        free(ctx);
        return NULL;
    }

    /* Register UMEM with the socket */
    struct xdp_umem_reg umem_reg;
    memset(&umem_reg, 0, sizeof(umem_reg));
    umem_reg.addr      = (uint64_t)(unsigned long)ctx->umem_area;
    umem_reg.len       = PION_XDP_UMEM_SIZE;
    umem_reg.chunk_size = PION_XDP_FRAME_SIZE;
    umem_reg.headroom  = 0;
    if (setsockopt(ctx->xsk_fd, SOL_XDP, XDP_UMEM_REG, &umem_reg, sizeof(umem_reg)) < 0) {
        fprintf(stderr, "XDP: UMEM registration failed (errno=%d)\n", errno);
        goto fail;
    }

    /* Set ring sizes */
    int ring_size = PION_XDP_RING_SIZE;
    setsockopt(ctx->xsk_fd, SOL_XDP, XDP_UMEM_FILL_RING,      &ring_size, sizeof(ring_size));
    setsockopt(ctx->xsk_fd, SOL_XDP, XDP_UMEM_COMPLETION_RING, &ring_size, sizeof(ring_size));
    setsockopt(ctx->xsk_fd, SOL_XDP, XDP_RX_RING,              &ring_size, sizeof(ring_size));
    setsockopt(ctx->xsk_fd, SOL_XDP, XDP_TX_RING,              &ring_size, sizeof(ring_size));

    /* Get ring offsets via getsockopt */
    struct xdp_mmap_offsets offsets;
    socklen_t optlen = sizeof(offsets);
    if (getsockopt(ctx->xsk_fd, SOL_XDP, XDP_MMAP_OFFSETS, &offsets, &optlen) < 0) {
        fprintf(stderr, "XDP: failed to get mmap offsets (errno=%d)\n", errno);
        goto fail;
    }

    /* mmap RX ring */
    void *rx_map = mmap(NULL,
        offsets.rx.desc + PION_XDP_RING_SIZE * sizeof(struct xdp_desc),
        PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE,
        ctx->xsk_fd, XDP_PGOFF_RX_RING);
    if (rx_map == MAP_FAILED) goto fail;
    ctx->rx_producer = (uint32_t *)((uint8_t *)rx_map + offsets.rx.producer);
    ctx->rx_consumer = (uint32_t *)((uint8_t *)rx_map + offsets.rx.consumer);
    ctx->rx_ring     = (struct xdp_desc *)((uint8_t *)rx_map + offsets.rx.desc);
    ctx->rx_mask     = PION_XDP_RING_SIZE - 1;

    /* mmap TX ring */
    void *tx_map = mmap(NULL,
        offsets.tx.desc + PION_XDP_RING_SIZE * sizeof(struct xdp_desc),
        PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE,
        ctx->xsk_fd, XDP_PGOFF_TX_RING);
    if (tx_map == MAP_FAILED) goto fail;
    ctx->tx_producer = (uint32_t *)((uint8_t *)tx_map + offsets.tx.producer);
    ctx->tx_consumer = (uint32_t *)((uint8_t *)tx_map + offsets.tx.consumer);
    ctx->tx_ring     = (struct xdp_desc *)((uint8_t *)tx_map + offsets.tx.desc);
    ctx->tx_mask     = PION_XDP_RING_SIZE - 1;

    /* mmap Fill ring */
    void *fill_map = mmap(NULL,
        offsets.fr.desc + PION_XDP_RING_SIZE * sizeof(uint64_t),
        PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE,
        ctx->xsk_fd, XDP_UMEM_PGOFF_FILL_RING);
    if (fill_map == MAP_FAILED) goto fail;
    ctx->fill_producer = (uint32_t *)((uint8_t *)fill_map + offsets.fr.producer);
    ctx->fill_consumer = (uint32_t *)((uint8_t *)fill_map + offsets.fr.consumer);
    ctx->fill_ring     = (uint64_t *)((uint8_t *)fill_map + offsets.fr.desc);
    ctx->fill_mask     = PION_XDP_RING_SIZE - 1;

    /* mmap Completion ring */
    void *comp_map = mmap(NULL,
        offsets.cr.desc + PION_XDP_RING_SIZE * sizeof(uint64_t),
        PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE,
        ctx->xsk_fd, XDP_UMEM_PGOFF_COMPLETION_RING);
    if (comp_map == MAP_FAILED) goto fail;
    ctx->comp_producer = (uint32_t *)((uint8_t *)comp_map + offsets.cr.producer);
    ctx->comp_consumer = (uint32_t *)((uint8_t *)comp_map + offsets.cr.consumer);
    ctx->comp_ring     = (uint64_t *)((uint8_t *)comp_map + offsets.cr.desc);
    ctx->comp_mask     = PION_XDP_RING_SIZE - 1;

    /* Pre-populate fill ring with empty frames for initial RX */
    _xdp_populate_fill_ring(ctx, PION_XDP_RING_SIZE / 2);

    /* Create XSKMAP for BPF → AF_XDP socket redirection */
    {
        union bpf_attr map_attr;
        memset(&map_attr, 0, sizeof(map_attr));
        map_attr.map_type    = BPF_MAP_TYPE_XSKMAP;
        map_attr.key_size    = 4;
        map_attr.value_size  = 4;
        map_attr.max_entries = 64;
        ctx->xskmap_fd = bpf_syscall(BPF_MAP_CREATE, &map_attr, sizeof(map_attr));
        if (ctx->xskmap_fd < 0) {
            fprintf(stderr, "XDP: XSKMAP create failed (errno=%d)\n", errno);
            goto fail;
        }
    }

    /* Load BPF program (references the XSKMAP) */
    uint16_t port_be = htons(target_port);
    ctx->bpf_prog_fd = _xdp_load_bpf_prog(port_be, ctx->xskmap_fd);
    if (ctx->bpf_prog_fd < 0) {
        fprintf(stderr, "XDP: BPF program load failed. Falling back to io_uring.\n");
        goto fail;
    }

    /* Try native/driver mode first */
    union bpf_attr attach_attr;
    memset(&attach_attr, 0, sizeof(attach_attr));
    /* Use NETLINK approach: set via setsockopt on raw socket */
    /* Simpler: use if_xdp approach with bpf_set_link_xdp_fd equivalent */
    /* For maximum compatibility, use the bpf(BPF_LINK_CREATE) syscall on 5.9+ */

    /* Bind AF_XDP socket to the interface + queue */
    struct sockaddr_xdp sxdp;
    memset(&sxdp, 0, sizeof(sxdp));
    sxdp.sxdp_family   = AF_XDP;
    sxdp.sxdp_ifindex  = ctx->ifindex;
    sxdp.sxdp_queue_id = queue_id;
    sxdp.sxdp_flags    = XDP_ZEROCOPY | XDP_USE_NEED_WAKEUP;

    if (bind(ctx->xsk_fd, (struct sockaddr *)&sxdp, sizeof(sxdp)) < 0) {
        /* Zero-copy not supported — try copy mode */
        sxdp.sxdp_flags = XDP_COPY | XDP_USE_NEED_WAKEUP;
        if (bind(ctx->xsk_fd, (struct sockaddr *)&sxdp, sizeof(sxdp)) < 0) {
            fprintf(stderr, "XDP: bind failed (errno=%d). Interface may not support AF_XDP.\n", errno);
            goto fail;
        }
        fprintf(stderr, "XDP: using copy mode (zero-copy not available on %s)\n", ifname);
    }

    /* Register AF_XDP socket in XSKMAP so BPF bpf_redirect_map() can find it.
     * Key = queue_id, Value = xsk_fd. */
    {
        union bpf_attr update_attr;
        memset(&update_attr, 0, sizeof(update_attr));
        update_attr.map_fd = ctx->xskmap_fd;
        update_attr.key    = (uint64_t)(unsigned long)&queue_id;
        update_attr.value  = (uint64_t)(unsigned long)&ctx->xsk_fd;
        update_attr.flags  = 0;  /* BPF_ANY */
        if (bpf_syscall(BPF_MAP_UPDATE_ELEM, &update_attr, sizeof(update_attr)) < 0) {
            fprintf(stderr, "XDP: XSKMAP update failed (errno=%d)\n", errno);
            goto fail;
        }
        printf("XDP: socket fd=%d registered in XSKMAP at key=%d\n", ctx->xsk_fd, queue_id);
    }

    /* Attach XDP program to NIC via netlink (IFLA_XDP_FD).
     * This is the standard way without libbpf. */
    {
        struct {
            struct nlmsghdr  nlh;
            struct ifinfomsg ifi;
            char             attrbuf[256];
        } req;
        memset(&req, 0, sizeof(req));
        req.nlh.nlmsg_len   = NLMSG_LENGTH(sizeof(struct ifinfomsg));
        req.nlh.nlmsg_type  = RTM_SETLINK;
        req.nlh.nlmsg_flags = NLM_F_REQUEST | NLM_F_ACK;
        req.ifi.ifi_family  = AF_UNSPEC;
        req.ifi.ifi_index   = ctx->ifindex;

        /* NLA: IFLA_XDP (nested), containing IFLA_XDP_FD */
        struct nlattr *nla_xdp = (struct nlattr *)((char *)&req + req.nlh.nlmsg_len);
        nla_xdp->nla_type = 43 | 0x8000;  /* IFLA_XDP | NLA_F_NESTED */
        struct nlattr *nla_fd = (struct nlattr *)((char *)nla_xdp + NLA_HDRLEN);
        nla_fd->nla_type = 1;  /* IFLA_XDP_FD */
        nla_fd->nla_len  = NLA_HDRLEN + sizeof(int);
        memcpy((char *)nla_fd + NLA_HDRLEN, &ctx->bpf_prog_fd, sizeof(int));
        nla_xdp->nla_len = NLA_HDRLEN + nla_fd->nla_len;

        /* Add flags (try SKB mode for veth, driver mode for real NICs) */
        struct nlattr *nla_flags = (struct nlattr *)((char *)nla_fd + NLA_ALIGN(nla_fd->nla_len));
        nla_flags->nla_type = 3;  /* IFLA_XDP_FLAGS */
        nla_flags->nla_len  = NLA_HDRLEN + sizeof(uint32_t);
        uint32_t xdp_flags = 0;  /* 0 = auto (kernel picks best: native if supported, generic fallback) */
        memcpy((char *)nla_flags + NLA_HDRLEN, &xdp_flags, sizeof(xdp_flags));
        nla_xdp->nla_len += NLA_ALIGN(nla_flags->nla_len);

        req.nlh.nlmsg_len += NLA_ALIGN(nla_xdp->nla_len);

        int nl_fd = socket(AF_NETLINK, SOCK_RAW, NETLINK_ROUTE);
        if (nl_fd < 0) {
            fprintf(stderr, "XDP: netlink socket failed (errno=%d)\n", errno);
            goto fail;
        }
        if (send(nl_fd, &req, req.nlh.nlmsg_len, 0) < 0) {
            fprintf(stderr, "XDP: netlink send failed (errno=%d)\n", errno);
            close(nl_fd);
            goto fail;
        }
        /* Read ACK */
        char resp[4096];
        int rlen = recv(nl_fd, resp, sizeof(resp), 0);
        close(nl_fd);
        if (rlen > 0) {
            struct nlmsghdr *nlh = (struct nlmsghdr *)resp;
            if (nlh->nlmsg_type == NLMSG_ERROR) {
                int *err = (int *)NLMSG_DATA(nlh);
                if (*err < 0) {
                    fprintf(stderr, "XDP: netlink attach failed (err=%d)\n", *err);
                    goto fail;
                }
            }
        }
        printf("XDP: program attached to %s (SKB mode)\n", ifname);
    }

    printf("XDP: AF_XDP socket created on %s queue=%d port=%d (fd=%d)\n",
           ifname, queue_id, target_port, ctx->xsk_fd);
    return ctx;

fail:
    if (ctx->xsk_fd >= 0) close(ctx->xsk_fd);
    if (ctx->bpf_prog_fd >= 0) close(ctx->bpf_prog_fd);
    if (ctx->xskmap_fd >= 0) close(ctx->xskmap_fd);
    if (ctx->umem_area && ctx->umem_area != MAP_FAILED)
        munmap(ctx->umem_area, PION_XDP_UMEM_SIZE);
    free(ctx);
    return NULL;
}

/* Poll RX ring for received frames. Returns number of frames available.
 * Fills out_addrs[] with UMEM addresses and out_lens[] with frame lengths.
 * Caller must call pion_xdp_rx_release() after processing each frame. */
int pion_xdp_poll_rx(void *handle, uint64_t *out_addrs, uint32_t *out_lens, int max_frames) {
    PionXDPContext *ctx = (PionXDPContext *)handle;
    if (!ctx) return 0;

    uint32_t prod = __atomic_load_n(ctx->rx_producer, __ATOMIC_ACQUIRE);
    uint32_t cons = *ctx->rx_consumer;
    uint32_t avail = prod - cons;
    if (avail == 0) return 0;
    if ((int)avail > max_frames) avail = (uint32_t)max_frames;

    for (uint32_t i = 0; i < avail; i++) {
        uint32_t idx = (cons + i) & ctx->rx_mask;
        out_addrs[i] = ctx->rx_ring[idx].addr;
        out_lens[i]  = ctx->rx_ring[idx].len;
    }

    __sync_synchronize();
    *ctx->rx_consumer = cons + avail;
    ctx->rx_packets += avail;

    /* Refill the fill ring with the same number of consumed frames */
    _xdp_populate_fill_ring(ctx, avail);

    return (int)avail;
}

/* Get pointer to a frame's data in UMEM given its address. */
void *pion_xdp_frame_ptr(void *handle, uint64_t addr) {
    PionXDPContext *ctx = (PionXDPContext *)handle;
    return (uint8_t *)ctx->umem_area + addr;
}

/* Release a received frame back to the free pool. */
void pion_xdp_rx_release(void *handle, uint64_t addr) {
    PionXDPContext *ctx = (PionXDPContext *)handle;
    _xdp_free_frame(ctx, addr);
}

/* Submit a frame for transmission via the TX ring.
 * The frame data must already be written into the UMEM area at the given address.
 * Returns 0 on success, -1 if TX ring is full. */
int pion_xdp_submit_tx(void *handle, uint64_t addr, uint32_t len) {
    PionXDPContext *ctx = (PionXDPContext *)handle;
    uint32_t prod = *ctx->tx_producer;
    uint32_t cons = __atomic_load_n(ctx->tx_consumer, __ATOMIC_ACQUIRE);
    if (prod - cons >= PION_XDP_RING_SIZE) return -1;  /* TX ring full */

    uint32_t idx = prod & ctx->tx_mask;
    ctx->tx_ring[idx].addr = addr;
    ctx->tx_ring[idx].len  = len;

    __sync_synchronize();
    *ctx->tx_producer = prod + 1;
    ctx->tx_packets++;
    return 0;
}

/* Kick the kernel to process pending TX submissions.
 * On AF_XDP, this is done via sendto() on the socket. */
int pion_xdp_tx_kick(void *handle) {
    PionXDPContext *ctx = (PionXDPContext *)handle;
    return (int)sendto(ctx->xsk_fd, NULL, 0, MSG_DONTWAIT, NULL, 0);
}

/* Drain the completion ring (TX frames the kernel has finished sending).
 * Returns freed frame addresses to the free pool. */
int pion_xdp_drain_completion(void *handle) {
    PionXDPContext *ctx = (PionXDPContext *)handle;
    uint32_t prod = __atomic_load_n(ctx->comp_producer, __ATOMIC_ACQUIRE);
    uint32_t cons = *ctx->comp_consumer;
    uint32_t avail = prod - cons;
    if (avail == 0) return 0;

    for (uint32_t i = 0; i < avail; i++) {
        uint32_t idx = (cons + i) & ctx->comp_mask;
        _xdp_free_frame(ctx, ctx->comp_ring[idx]);
    }

    __sync_synchronize();
    *ctx->comp_consumer = cons + avail;
    return (int)avail;
}

/* Get the AF_XDP socket fd for use with poll()/epoll(). */
int pion_xdp_get_fd(void *handle) {
    PionXDPContext *ctx = (PionXDPContext *)handle;
    return ctx ? ctx->xsk_fd : -1;
}

/* Get UMEM base pointer for direct access from Mojo. */
void *pion_xdp_get_umem(void *handle) {
    PionXDPContext *ctx = (PionXDPContext *)handle;
    return ctx ? ctx->umem_area : NULL;
}

/* Allocate a TX frame from the free pool. Returns UMEM address or UINT64_MAX if none. */
uint64_t pion_xdp_alloc_tx_frame(void *handle) {
    PionXDPContext *ctx = (PionXDPContext *)handle;
    return _xdp_alloc_frame(ctx);
}

/* Get statistics */
void pion_xdp_stats(void *handle, uint64_t *rx_pkts, uint64_t *tx_pkts,
                     uint64_t *rx_bytes, uint64_t *tx_bytes) {
    PionXDPContext *ctx = (PionXDPContext *)handle;
    if (!ctx) return;
    *rx_pkts  = ctx->rx_packets;
    *tx_pkts  = ctx->tx_packets;
    *rx_bytes = ctx->rx_bytes;
    *tx_bytes = ctx->tx_bytes;
}

/* Detach XDP program from NIC via netlink (IFLA_XDP_FD = -1).
 * Called on cleanup to ensure BPF program doesn't stay attached after exit. */
void pion_xdp_detach(void *handle) {
    PionXDPContext *ctx = (PionXDPContext *)handle;
    if (!ctx || ctx->ifindex <= 0) return;

    struct {
        struct nlmsghdr  nlh;
        struct ifinfomsg ifi;
        char             attrbuf[256];
    } req;
    memset(&req, 0, sizeof(req));
    req.nlh.nlmsg_len   = NLMSG_LENGTH(sizeof(struct ifinfomsg));
    req.nlh.nlmsg_type  = RTM_SETLINK;
    req.nlh.nlmsg_flags = NLM_F_REQUEST | NLM_F_ACK;
    req.ifi.ifi_family  = AF_UNSPEC;
    req.ifi.ifi_index   = ctx->ifindex;

    struct nlattr *nla_xdp = (struct nlattr *)((char *)&req + req.nlh.nlmsg_len);
    nla_xdp->nla_type = 43 | 0x8000;  /* IFLA_XDP | NLA_F_NESTED */
    struct nlattr *nla_fd = (struct nlattr *)((char *)nla_xdp + NLA_HDRLEN);
    nla_fd->nla_type = 1;  /* IFLA_XDP_FD */
    nla_fd->nla_len  = NLA_HDRLEN + sizeof(int);
    int detach_fd = -1;
    memcpy((char *)nla_fd + NLA_HDRLEN, &detach_fd, sizeof(int));
    nla_xdp->nla_len = NLA_HDRLEN + nla_fd->nla_len;
    req.nlh.nlmsg_len += NLA_ALIGN(nla_xdp->nla_len);

    int nl_fd = socket(AF_NETLINK, SOCK_RAW, NETLINK_ROUTE);
    if (nl_fd >= 0) {
        send(nl_fd, &req, req.nlh.nlmsg_len, 0);
        char resp[256];
        recv(nl_fd, resp, sizeof(resp), 0);
        close(nl_fd);
        printf("XDP: BPF program detached from ifindex=%d\n", ctx->ifindex);
    }
}

/* Send a large TCP response fragmented across multiple UMEM TX frames.
 * Each frame carries up to MAX_TCP_PAYLOAD bytes of payload data.
 * Returns total bytes sent (sum of all fragment payloads), or -1 on error.
 *
 * This fixes the P>1 pipeline bug where responses > 4KB were truncated to a
 * single frame. Now memtier can pipeline multiple commands and receive full
 * multi-frame responses. */
#define MAX_TCP_PAYLOAD (PION_XDP_FRAME_SIZE - 14 - 20 - 20)  /* 4042 bytes */

int pion_xdp_send_fragmented(void *handle, const void *src_frame, int src_frame_len,
                              uint8_t tcp_flags, uint32_t seq_num, uint32_t ack_num,
                              const void *payload, int payload_len)
{
    PionXDPContext *ctx = (PionXDPContext *)handle;
    if (!ctx || !payload || payload_len <= 0) return -1;

    int total_sent = 0;
    const uint8_t *data = (const uint8_t *)payload;
    int remaining = payload_len;

    while (remaining > 0) {
        int chunk = remaining > MAX_TCP_PAYLOAD ? MAX_TCP_PAYLOAD : remaining;

        uint64_t tx_addr = _xdp_alloc_frame(ctx);
        if (tx_addr == UINT64_MAX) {
            /* TX frame pool exhausted — kick pending frames and drain completions,
             * then retry once. This handles bursts where we've submitted many frames
             * but the kernel hasn't reclaimed them yet. */
            sendto(ctx->xsk_fd, NULL, 0, MSG_DONTWAIT, NULL, 0);

            /* Drain completion ring */
            uint32_t cprod = __atomic_load_n(ctx->comp_producer, __ATOMIC_ACQUIRE);
            uint32_t ccons = *ctx->comp_consumer;
            uint32_t cavail = cprod - ccons;
            for (uint32_t ci = 0; ci < cavail; ci++) {
                uint32_t cidx = (ccons + ci) & ctx->comp_mask;
                _xdp_free_frame(ctx, ctx->comp_ring[cidx]);
            }
            if (cavail > 0) {
                __sync_synchronize();
                *ctx->comp_consumer = ccons + cavail;
            }

            tx_addr = _xdp_alloc_frame(ctx);
            if (tx_addr == UINT64_MAX) {
                fprintf(stderr, "XDP: TX frame pool exhausted, %d/%d bytes sent\n", total_sent, payload_len);
                break;  /* partial send — return what we managed */
            }
        }

        uint8_t *tx_buf = (uint8_t *)ctx->umem_area + tx_addr;

        /* Use PSH+ACK on last fragment, ACK on intermediate fragments */
        uint8_t frag_flags = (remaining <= MAX_TCP_PAYLOAD) ? tcp_flags : (tcp_flags & ~0x08);  /* clear PSH on non-last */

        int frame_len = pion_xdp_build_tcp_response(
            tx_buf, src_frame, src_frame_len,
            frag_flags, seq_num + total_sent, ack_num,
            data + total_sent, chunk
        );
        if (frame_len <= 0) {
            _xdp_free_frame(ctx, tx_addr);
            break;
        }

        /* Submit to TX ring */
        uint32_t prod = *ctx->tx_producer;
        uint32_t cons = __atomic_load_n(ctx->tx_consumer, __ATOMIC_ACQUIRE);
        if (prod - cons >= PION_XDP_RING_SIZE) {
            /* TX ring full — kick and retry once */
            sendto(ctx->xsk_fd, NULL, 0, MSG_DONTWAIT, NULL, 0);
            prod = *ctx->tx_producer;
            cons = __atomic_load_n(ctx->tx_consumer, __ATOMIC_ACQUIRE);
            if (prod - cons >= PION_XDP_RING_SIZE) {
                _xdp_free_frame(ctx, tx_addr);
                fprintf(stderr, "XDP: TX ring full, %d/%d bytes sent\n", total_sent, payload_len);
                break;
            }
        }

        uint32_t idx = prod & ctx->tx_mask;
        ctx->tx_ring[idx].addr = tx_addr;
        ctx->tx_ring[idx].len  = (uint32_t)frame_len;
        __sync_synchronize();
        *ctx->tx_producer = prod + 1;
        ctx->tx_packets++;

        total_sent += chunk;
        remaining -= chunk;

        /* Kick periodically during large sends to avoid ring exhaustion.
         * Every 32 frames (~128KB), kick the kernel to start sending. */
        if ((total_sent / MAX_TCP_PAYLOAD) % 32 == 0 && remaining > 0) {
            sendto(ctx->xsk_fd, NULL, 0, MSG_DONTWAIT, NULL, 0);
        }
    }

    return total_sent > 0 ? total_sent : -1;
}

/* Create a shared XSKMAP before parallelize. Returns xskmap_fd or -1.
 * Used by multi-worker XDP: one BPF program + one XSKMAP shared across all workers. */
int pion_xdp_create_shared_xskmap(int max_queues) {
    union bpf_attr map_attr;
    memset(&map_attr, 0, sizeof(map_attr));
    map_attr.map_type    = BPF_MAP_TYPE_XSKMAP;
    map_attr.key_size    = 4;
    map_attr.value_size  = 4;
    map_attr.max_entries = max_queues > 0 ? max_queues : 64;
    int fd = bpf_syscall(BPF_MAP_CREATE, &map_attr, sizeof(map_attr));
    if (fd < 0) {
        fprintf(stderr, "XDP: shared XSKMAP create failed (errno=%d)\n", errno);
    }
    return fd;
}

/* Load BPF program referencing a pre-created XSKMAP. Returns prog_fd or -1. */
int pion_xdp_load_shared_bpf(uint16_t target_port, int xskmap_fd) {
    uint16_t port_be = htons(target_port);
    return _xdp_load_bpf_prog(port_be, xskmap_fd);
}

/* Attach BPF program to NIC. Returns 0 on success, -1 on failure. */
int pion_xdp_attach_bpf(const char *ifname, int bpf_prog_fd) {
    int ifindex = (int)if_nametoindex(ifname);
    if (ifindex == 0) {
        fprintf(stderr, "XDP: interface '%s' not found\n", ifname);
        return -1;
    }

    struct {
        struct nlmsghdr  nlh;
        struct ifinfomsg ifi;
        char             attrbuf[256];
    } req;
    memset(&req, 0, sizeof(req));
    req.nlh.nlmsg_len   = NLMSG_LENGTH(sizeof(struct ifinfomsg));
    req.nlh.nlmsg_type  = RTM_SETLINK;
    req.nlh.nlmsg_flags = NLM_F_REQUEST | NLM_F_ACK;
    req.ifi.ifi_family  = AF_UNSPEC;
    req.ifi.ifi_index   = ifindex;

    struct nlattr *nla_xdp = (struct nlattr *)((char *)&req + req.nlh.nlmsg_len);
    nla_xdp->nla_type = 43 | 0x8000;  /* IFLA_XDP | NLA_F_NESTED */
    struct nlattr *nla_fd = (struct nlattr *)((char *)nla_xdp + NLA_HDRLEN);
    nla_fd->nla_type = 1;  /* IFLA_XDP_FD */
    nla_fd->nla_len  = NLA_HDRLEN + sizeof(int);
    memcpy((char *)nla_fd + NLA_HDRLEN, &bpf_prog_fd, sizeof(int));
    nla_xdp->nla_len = NLA_HDRLEN + nla_fd->nla_len;

    struct nlattr *nla_flags = (struct nlattr *)((char *)nla_fd + NLA_ALIGN(nla_fd->nla_len));
    nla_flags->nla_type = 3;  /* IFLA_XDP_FLAGS */
    nla_flags->nla_len  = NLA_HDRLEN + sizeof(uint32_t);
    uint32_t xdp_flags = 0;  /* auto mode */
    memcpy((char *)nla_flags + NLA_HDRLEN, &xdp_flags, sizeof(xdp_flags));
    nla_xdp->nla_len += NLA_ALIGN(nla_flags->nla_len);
    req.nlh.nlmsg_len += NLA_ALIGN(nla_xdp->nla_len);

    int nl_fd = socket(AF_NETLINK, SOCK_RAW, NETLINK_ROUTE);
    if (nl_fd < 0) return -1;

    if (send(nl_fd, &req, req.nlh.nlmsg_len, 0) < 0) {
        close(nl_fd);
        return -1;
    }
    char resp[4096];
    int rlen = recv(nl_fd, resp, sizeof(resp), 0);
    close(nl_fd);

    if (rlen > 0) {
        struct nlmsghdr *nlh = (struct nlmsghdr *)resp;
        if (nlh->nlmsg_type == NLMSG_ERROR) {
            int *err = (int *)NLMSG_DATA(nlh);
            if (*err < 0) {
                fprintf(stderr, "XDP: netlink attach failed (err=%d)\n", *err);
                return -1;
            }
        }
    }
    printf("XDP: BPF program attached to %s\n", ifname);
    return 0;
}

/* Detach XDP program from NIC by interface name. */
void pion_xdp_detach_by_name(const char *ifname) {
    int ifindex = (int)if_nametoindex(ifname);
    if (ifindex == 0) return;

    struct {
        struct nlmsghdr  nlh;
        struct ifinfomsg ifi;
        char             attrbuf[256];
    } req;
    memset(&req, 0, sizeof(req));
    req.nlh.nlmsg_len   = NLMSG_LENGTH(sizeof(struct ifinfomsg));
    req.nlh.nlmsg_type  = RTM_SETLINK;
    req.nlh.nlmsg_flags = NLM_F_REQUEST | NLM_F_ACK;
    req.ifi.ifi_family  = AF_UNSPEC;
    req.ifi.ifi_index   = ifindex;

    struct nlattr *nla_xdp = (struct nlattr *)((char *)&req + req.nlh.nlmsg_len);
    nla_xdp->nla_type = 43 | 0x8000;
    struct nlattr *nla_fd = (struct nlattr *)((char *)nla_xdp + NLA_HDRLEN);
    nla_fd->nla_type = 1;
    nla_fd->nla_len  = NLA_HDRLEN + sizeof(int);
    int detach_fd = -1;
    memcpy((char *)nla_fd + NLA_HDRLEN, &detach_fd, sizeof(int));
    nla_xdp->nla_len = NLA_HDRLEN + nla_fd->nla_len;
    req.nlh.nlmsg_len += NLA_ALIGN(nla_xdp->nla_len);

    int nl_fd = socket(AF_NETLINK, SOCK_RAW, NETLINK_ROUTE);
    if (nl_fd >= 0) {
        send(nl_fd, &req, req.nlh.nlmsg_len, 0);
        char resp[256];
        recv(nl_fd, resp, sizeof(resp), 0);
        close(nl_fd);
        printf("XDP: BPF program detached from %s\n", ifname);
    }
}

/* Create AF_XDP socket for a worker using a pre-existing shared XSKMAP.
 * Does NOT load BPF or attach — assumes pion_xdp_attach_bpf() already called.
 * Returns context handle, or NULL on failure. */
void *pion_xdp_create_worker(const char *ifname, int queue_id, uint16_t target_port,
                              int shared_xskmap_fd, int shared_bpf_prog_fd) {
    PionXDPContext *ctx = (PionXDPContext *)calloc(1, sizeof(PionXDPContext));
    if (!ctx) return NULL;

    ctx->ifindex = (int)if_nametoindex(ifname);
    if (ctx->ifindex == 0) {
        fprintf(stderr, "XDP: interface '%s' not found\n", ifname);
        free(ctx);
        return NULL;
    }
    ctx->queue_id = queue_id;
    ctx->target_port = target_port;
    ctx->xsk_fd = -1;
    ctx->bpf_prog_fd = shared_bpf_prog_fd;  /* shared, don't close on destroy */
    ctx->xskmap_fd = shared_xskmap_fd;       /* shared, don't close on destroy */

    /* Allocate UMEM */
    ctx->umem_area = mmap(NULL, PION_XDP_UMEM_SIZE, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
    if (ctx->umem_area == MAP_FAILED) {
        free(ctx);
        return NULL;
    }
    ctx->umem_size = PION_XDP_UMEM_SIZE;

    ctx->free_frame_count = PION_XDP_NUM_FRAMES;
    for (uint32_t i = 0; i < PION_XDP_NUM_FRAMES; i++)
        ctx->free_frames[i] = i * PION_XDP_FRAME_SIZE;

    /* Create AF_XDP socket */
    ctx->xsk_fd = socket(AF_XDP, SOCK_RAW, 0);
    if (ctx->xsk_fd < 0) {
        munmap(ctx->umem_area, PION_XDP_UMEM_SIZE);
        free(ctx);
        return NULL;
    }

    /* Register UMEM */
    struct xdp_umem_reg umem_reg;
    memset(&umem_reg, 0, sizeof(umem_reg));
    umem_reg.addr      = (uint64_t)(unsigned long)ctx->umem_area;
    umem_reg.len       = PION_XDP_UMEM_SIZE;
    umem_reg.chunk_size = PION_XDP_FRAME_SIZE;
    umem_reg.headroom  = 0;
    if (setsockopt(ctx->xsk_fd, SOL_XDP, XDP_UMEM_REG, &umem_reg, sizeof(umem_reg)) < 0) {
        fprintf(stderr, "XDP worker %d: UMEM registration failed (errno=%d)\n", queue_id, errno);
        goto fail_worker;
    }

    /* Set ring sizes */
    int ring_size = PION_XDP_RING_SIZE;
    setsockopt(ctx->xsk_fd, SOL_XDP, XDP_UMEM_FILL_RING,      &ring_size, sizeof(ring_size));
    setsockopt(ctx->xsk_fd, SOL_XDP, XDP_UMEM_COMPLETION_RING, &ring_size, sizeof(ring_size));
    setsockopt(ctx->xsk_fd, SOL_XDP, XDP_RX_RING,              &ring_size, sizeof(ring_size));
    setsockopt(ctx->xsk_fd, SOL_XDP, XDP_TX_RING,              &ring_size, sizeof(ring_size));

    /* Get ring offsets */
    struct xdp_mmap_offsets offsets;
    socklen_t optlen = sizeof(offsets);
    if (getsockopt(ctx->xsk_fd, SOL_XDP, XDP_MMAP_OFFSETS, &offsets, &optlen) < 0)
        goto fail_worker;

    /* mmap all 4 rings */
    void *rx_map = mmap(NULL, offsets.rx.desc + PION_XDP_RING_SIZE * sizeof(struct xdp_desc),
                        PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ctx->xsk_fd, XDP_PGOFF_RX_RING);
    if (rx_map == MAP_FAILED) goto fail_worker;
    ctx->rx_producer = (uint32_t *)((uint8_t *)rx_map + offsets.rx.producer);
    ctx->rx_consumer = (uint32_t *)((uint8_t *)rx_map + offsets.rx.consumer);
    ctx->rx_ring     = (struct xdp_desc *)((uint8_t *)rx_map + offsets.rx.desc);
    ctx->rx_mask     = PION_XDP_RING_SIZE - 1;

    void *tx_map = mmap(NULL, offsets.tx.desc + PION_XDP_RING_SIZE * sizeof(struct xdp_desc),
                        PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ctx->xsk_fd, XDP_PGOFF_TX_RING);
    if (tx_map == MAP_FAILED) goto fail_worker;
    ctx->tx_producer = (uint32_t *)((uint8_t *)tx_map + offsets.tx.producer);
    ctx->tx_consumer = (uint32_t *)((uint8_t *)tx_map + offsets.tx.consumer);
    ctx->tx_ring     = (struct xdp_desc *)((uint8_t *)tx_map + offsets.tx.desc);
    ctx->tx_mask     = PION_XDP_RING_SIZE - 1;

    void *fill_map = mmap(NULL, offsets.fr.desc + PION_XDP_RING_SIZE * sizeof(uint64_t),
                          PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ctx->xsk_fd, XDP_UMEM_PGOFF_FILL_RING);
    if (fill_map == MAP_FAILED) goto fail_worker;
    ctx->fill_producer = (uint32_t *)((uint8_t *)fill_map + offsets.fr.producer);
    ctx->fill_consumer = (uint32_t *)((uint8_t *)fill_map + offsets.fr.consumer);
    ctx->fill_ring     = (uint64_t *)((uint8_t *)fill_map + offsets.fr.desc);
    ctx->fill_mask     = PION_XDP_RING_SIZE - 1;

    void *comp_map = mmap(NULL, offsets.cr.desc + PION_XDP_RING_SIZE * sizeof(uint64_t),
                          PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ctx->xsk_fd, XDP_UMEM_PGOFF_COMPLETION_RING);
    if (comp_map == MAP_FAILED) goto fail_worker;
    ctx->comp_producer = (uint32_t *)((uint8_t *)comp_map + offsets.cr.producer);
    ctx->comp_consumer = (uint32_t *)((uint8_t *)comp_map + offsets.cr.consumer);
    ctx->comp_ring     = (uint64_t *)((uint8_t *)comp_map + offsets.cr.desc);
    ctx->comp_mask     = PION_XDP_RING_SIZE - 1;

    /* Pre-populate fill ring */
    _xdp_populate_fill_ring(ctx, PION_XDP_RING_SIZE / 2);

    /* Bind to interface + queue */
    struct sockaddr_xdp sxdp;
    memset(&sxdp, 0, sizeof(sxdp));
    sxdp.sxdp_family   = AF_XDP;
    sxdp.sxdp_ifindex  = ctx->ifindex;
    sxdp.sxdp_queue_id = queue_id;
    sxdp.sxdp_flags    = XDP_ZEROCOPY | XDP_USE_NEED_WAKEUP;
    if (bind(ctx->xsk_fd, (struct sockaddr *)&sxdp, sizeof(sxdp)) < 0) {
        sxdp.sxdp_flags = XDP_COPY | XDP_USE_NEED_WAKEUP;
        if (bind(ctx->xsk_fd, (struct sockaddr *)&sxdp, sizeof(sxdp)) < 0) {
            fprintf(stderr, "XDP worker %d: bind failed (errno=%d)\n", queue_id, errno);
            goto fail_worker;
        }
    }

    /* Register this worker's socket in the SHARED XSKMAP */
    {
        union bpf_attr update_attr;
        memset(&update_attr, 0, sizeof(update_attr));
        update_attr.map_fd = shared_xskmap_fd;
        update_attr.key    = (uint64_t)(unsigned long)&queue_id;
        update_attr.value  = (uint64_t)(unsigned long)&ctx->xsk_fd;
        update_attr.flags  = 0;
        if (bpf_syscall(BPF_MAP_UPDATE_ELEM, &update_attr, sizeof(update_attr)) < 0) {
            fprintf(stderr, "XDP worker %d: XSKMAP update failed (errno=%d)\n", queue_id, errno);
            goto fail_worker;
        }
        printf("XDP: worker %d socket fd=%d registered in shared XSKMAP at key=%d\n",
               queue_id, ctx->xsk_fd, queue_id);
    }

    return ctx;

fail_worker:
    if (ctx->xsk_fd >= 0) close(ctx->xsk_fd);
    /* Don't close shared fds — they belong to the parent */
    ctx->bpf_prog_fd = -1;
    ctx->xskmap_fd = -1;
    if (ctx->umem_area && ctx->umem_area != MAP_FAILED)
        munmap(ctx->umem_area, PION_XDP_UMEM_SIZE);
    free(ctx);
    return NULL;
}

/* Destroy a worker XDP context. Does NOT close shared BPF/XSKMAP fds. */
void pion_xdp_destroy_worker(void *handle) {
    PionXDPContext *ctx = (PionXDPContext *)handle;
    if (!ctx) return;
    if (ctx->xsk_fd >= 0) close(ctx->xsk_fd);
    /* Don't close bpf_prog_fd or xskmap_fd — they're shared */
    if (ctx->umem_area && ctx->umem_area != MAP_FAILED)
        munmap(ctx->umem_area, PION_XDP_UMEM_SIZE);
    free(ctx);
}

/* Set up flow steering: direct TCP traffic for target_port to a specific RX queue.
 * Uses ethtool SIOCETHTOOL ioctl with ETHTOOL_SRXCLSRLINS (add flow rule).
 * Returns 0 on success, -1 on failure. Falls back to shell ethtool if ioctl fails. */
int pion_xdp_setup_flow_steering(const char *ifname, uint16_t target_port, int queue_id) {
    /* Use system() as a reliable cross-driver approach — ethtool ioctls vary by driver.
     * The ethtool -N command uses ETHTOOL_SRXCLSRLINS under the hood. */
    char cmd[256];
    snprintf(cmd, sizeof(cmd),
             "ethtool -N %s flow-type tcp4 dst-port %d action %d 2>/dev/null",
             ifname, (int)target_port, queue_id);
    int ret = system(cmd);
    if (ret != 0) {
        fprintf(stderr, "XDP: flow steering setup failed for %s port %d → queue %d (ret=%d)\n",
                ifname, target_port, queue_id, ret);
        fprintf(stderr, "XDP: continuing without flow steering — packets may be distributed across queues\n");
        return -1;
    }
    printf("XDP: flow steering: %s port %d → RX queue %d\n", ifname, target_port, queue_id);
    return 0;
}

/* Remove flow steering rules for a port. */
void pion_xdp_remove_flow_steering(const char *ifname, uint16_t target_port) {
    /* List rules and delete matching ones */
    char cmd[256];
    snprintf(cmd, sizeof(cmd),
             "ethtool -n %s 2>/dev/null | grep -B1 'dst-port %d' | grep 'Filter:' | awk '{print $2}' | while read id; do ethtool -N %s delete $id 2>/dev/null; done",
             ifname, (int)target_port, ifname);
    system(cmd);
}

/* Clean up and destroy the XDP context. Detaches BPF from NIC. */
void pion_xdp_destroy(void *handle) {
    PionXDPContext *ctx = (PionXDPContext *)handle;
    if (!ctx) return;
    /* Detach BPF program from NIC before closing fds */
    pion_xdp_detach(handle);
    if (ctx->xsk_fd >= 0) close(ctx->xsk_fd);
    if (ctx->bpf_prog_fd >= 0) close(ctx->bpf_prog_fd);
    if (ctx->xskmap_fd >= 0) close(ctx->xskmap_fd);
    if (ctx->umem_area && ctx->umem_area != MAP_FAILED)
        munmap(ctx->umem_area, PION_XDP_UMEM_SIZE);
    free(ctx);
}

/* ── TCP-Lite Packet Helpers ──────────────────────────────────────────────── */

/* Internet checksum computation (RFC 1071) */
static uint16_t _ip_checksum(const void *data, int len) {
    const uint16_t *p = (const uint16_t *)data;
    uint32_t sum = 0;
    while (len > 1) { sum += *p++; len -= 2; }
    if (len) sum += *(const uint8_t *)p;
    while (sum >> 16) sum = (sum & 0xFFFF) + (sum >> 16);
    return (uint16_t)~sum;
}

/* TCP checksum (includes pseudo-header) */
static uint16_t _tcp_checksum(uint32_t saddr, uint32_t daddr,
                               const void *tcp_data, int tcp_len) {
    uint32_t sum = 0;
    /* Pseudo-header */
    sum += (saddr >> 16) & 0xFFFF;
    sum += saddr & 0xFFFF;
    sum += (daddr >> 16) & 0xFFFF;
    sum += daddr & 0xFFFF;
    sum += htons(6);       /* IPPROTO_TCP */
    sum += htons(tcp_len);
    /* TCP segment */
    const uint16_t *p = (const uint16_t *)tcp_data;
    int len = tcp_len;
    while (len > 1) { sum += *p++; len -= 2; }
    if (len) sum += *(const uint8_t *)p;
    while (sum >> 16) sum = (sum & 0xFFFF) + (sum >> 16);
    return (uint16_t)~sum;
}

/* Build a TCP response frame in the given UMEM buffer.
 * Swaps src/dst addresses from the incoming frame to create a response.
 * Returns total frame length (eth + ip + tcp + payload).
 *
 * Parameters:
 *   frame_buf:     destination buffer in UMEM
 *   src_frame:     incoming frame (for address/port extraction)
 *   src_frame_len: length of incoming frame
 *   tcp_flags:     TCP flags (SYN=0x02, ACK=0x10, SYN+ACK=0x12, FIN+ACK=0x11, RST=0x04)
 *   seq_num:       sequence number (network byte order)
 *   ack_num:       acknowledgment number (network byte order)
 *   payload:       TCP payload (NULL for control packets)
 *   payload_len:   payload length
 */
int pion_xdp_build_tcp_response(void *frame_buf, const void *src_frame,
                                 int src_frame_len,
                                 uint8_t tcp_flags,
                                 uint32_t seq_num, uint32_t ack_num,
                                 const void *payload, int payload_len)
{
    if (src_frame_len < 54) return -1;  /* eth(14) + ip(20) + tcp(20) minimum */

    const uint8_t *src = (const uint8_t *)src_frame;
    uint8_t *dst = (uint8_t *)frame_buf;

    /* Copy and swap Ethernet header (14 bytes) */
    memcpy(dst, src + 6, 6);      /* dst MAC = src MAC from incoming */
    memcpy(dst + 6, src, 6);      /* src MAC = dst MAC from incoming */
    dst[12] = src[12]; dst[13] = src[13];  /* ETH_P_IP */

    /* Build IPv4 header (20 bytes, no options) */
    const uint8_t *src_ip = src + 14;
    uint8_t *dst_ip = dst + 14;
    int tcp_total_len = 20 + payload_len;  /* TCP header (20, no options) + payload */
    int ip_total_len  = 20 + tcp_total_len;

    dst_ip[0] = 0x45;                    /* version=4, ihl=5 */
    dst_ip[1] = 0x00;                    /* TOS */
    dst_ip[2] = (ip_total_len >> 8) & 0xFF;
    dst_ip[3] = ip_total_len & 0xFF;
    dst_ip[4] = 0; dst_ip[5] = 0;       /* identification */
    dst_ip[6] = 0x40; dst_ip[7] = 0;    /* DF flag, frag offset 0 */
    dst_ip[8] = 64;                      /* TTL */
    dst_ip[9] = 6;                       /* TCP */
    dst_ip[10] = 0; dst_ip[11] = 0;     /* checksum (computed below) */
    /* Swap src/dst IP addresses */
    memcpy(dst_ip + 12, src_ip + 16, 4); /* our saddr = their daddr */
    memcpy(dst_ip + 16, src_ip + 12, 4); /* our daddr = their saddr */
    /* Compute IP checksum */
    uint16_t ip_cksum = _ip_checksum(dst_ip, 20);
    dst_ip[10] = ip_cksum & 0xFF;
    dst_ip[11] = (ip_cksum >> 8) & 0xFF;

    /* Build TCP header (20 bytes, no options for data/ack; 24 for SYN-ACK with MSS) */
    uint8_t *dst_tcp = dst + 34;
    /* Extract incoming TCP src/dst ports */
    uint8_t src_ihl = (src_ip[0] & 0x0F) * 4;
    const uint8_t *src_tcp = src + 14 + src_ihl;

    /* Swap src/dst ports */
    dst_tcp[0] = src_tcp[2]; dst_tcp[1] = src_tcp[3];  /* our sport = their dport */
    dst_tcp[2] = src_tcp[0]; dst_tcp[3] = src_tcp[1];  /* our dport = their sport */
    /* Sequence number */
    dst_tcp[4] = (seq_num >> 24) & 0xFF;
    dst_tcp[5] = (seq_num >> 16) & 0xFF;
    dst_tcp[6] = (seq_num >> 8) & 0xFF;
    dst_tcp[7] = seq_num & 0xFF;
    /* Ack number */
    dst_tcp[8]  = (ack_num >> 24) & 0xFF;
    dst_tcp[9]  = (ack_num >> 16) & 0xFF;
    dst_tcp[10] = (ack_num >> 8) & 0xFF;
    dst_tcp[11] = ack_num & 0xFF;
    /* Data offset (5 words = 20 bytes) | flags */
    dst_tcp[12] = 0x50;  /* data offset = 5 (20 bytes / 4) */
    dst_tcp[13] = tcp_flags;
    /* Window size: 65535 */
    dst_tcp[14] = 0xFF; dst_tcp[15] = 0xFF;
    /* Checksum (computed below) */
    dst_tcp[16] = 0; dst_tcp[17] = 0;
    /* Urgent pointer */
    dst_tcp[18] = 0; dst_tcp[19] = 0;

    /* Copy payload */
    if (payload && payload_len > 0) {
        memcpy(dst_tcp + 20, payload, payload_len);
    }

    /* Compute TCP checksum */
    uint32_t saddr, daddr;
    memcpy(&saddr, dst_ip + 12, 4);
    memcpy(&daddr, dst_ip + 16, 4);
    uint16_t tcp_cksum = _tcp_checksum(saddr, daddr, dst_tcp, tcp_total_len);
    dst_tcp[16] = tcp_cksum & 0xFF;
    dst_tcp[17] = (tcp_cksum >> 8) & 0xFF;

    return 14 + ip_total_len;  /* total frame length */
}

/* Extract TCP payload pointer and length from a raw Ethernet frame.
 * Returns payload length (>0 if data present, 0 for control packets, -1 on error).
 * Sets *out_payload to point into the frame data (zero-copy). */
int pion_xdp_extract_tcp_payload(const void *frame, int frame_len,
                                  const uint8_t **out_payload,
                                  uint32_t *out_seq, uint32_t *out_ack,
                                  uint8_t *out_flags,
                                  uint16_t *out_sport, uint16_t *out_dport)
{
    if (frame_len < 54) return -1;
    const uint8_t *p = (const uint8_t *)frame;

    /* Skip Ethernet (14 bytes) */
    uint16_t eth_proto = (p[12] << 8) | p[13];
    if (eth_proto != 0x0800) return -1;  /* not IPv4 */

    /* IP header */
    const uint8_t *ip = p + 14;
    uint8_t ihl = (ip[0] & 0x0F) * 4;
    if (ihl < 20 || ip[9] != 6) return -1;  /* not TCP */

    /* TCP header */
    const uint8_t *tcp = ip + ihl;
    if ((const uint8_t *)tcp + 20 > p + frame_len) return -1;

    *out_sport = (tcp[0] << 8) | tcp[1];
    *out_dport = (tcp[2] << 8) | tcp[3];
    *out_seq   = ((uint32_t)tcp[4] << 24) | ((uint32_t)tcp[5] << 16) |
                 ((uint32_t)tcp[6] << 8) | tcp[7];
    *out_ack   = ((uint32_t)tcp[8] << 24) | ((uint32_t)tcp[9] << 16) |
                 ((uint32_t)tcp[10] << 8) | tcp[11];
    *out_flags = tcp[13];

    uint8_t tcp_hlen = ((tcp[12] >> 4) & 0x0F) * 4;
    if (tcp_hlen < 20) return -1;

    int payload_offset = 14 + ihl + tcp_hlen;
    int payload_len = frame_len - payload_offset;
    if (payload_len < 0) payload_len = 0;

    *out_payload = (payload_len > 0) ? (p + payload_offset) : NULL;
    return payload_len;
}

#else  /* !__linux__ — macOS stubs */

void *pion_xdp_create(const char *ifname, int queue_id, uint16_t target_port) {
    (void)ifname; (void)queue_id; (void)target_port;
    fprintf(stderr, "XDP: not available on macOS (Linux 5.4+ required)\n");
    return NULL;
}
int   pion_xdp_poll_rx(void *h, uint64_t *a, uint32_t *l, int m) { (void)h;(void)a;(void)l;(void)m; return 0; }
void *pion_xdp_frame_ptr(void *h, uint64_t a) { (void)h;(void)a; return NULL; }
void  pion_xdp_rx_release(void *h, uint64_t a) { (void)h;(void)a; }
int   pion_xdp_submit_tx(void *h, uint64_t a, uint32_t l) { (void)h;(void)a;(void)l; return -1; }
int   pion_xdp_tx_kick(void *h) { (void)h; return -1; }
int   pion_xdp_drain_completion(void *h) { (void)h; return 0; }
int   pion_xdp_get_fd(void *h) { (void)h; return -1; }
void *pion_xdp_get_umem(void *h) { (void)h; return NULL; }
uint64_t pion_xdp_alloc_tx_frame(void *h) { (void)h; return (uint64_t)-1; }
void  pion_xdp_stats(void *h, uint64_t *a, uint64_t *b, uint64_t *c, uint64_t *d) { (void)h;(void)a;(void)b;(void)c;(void)d; }
void  pion_xdp_destroy(void *h) { (void)h; }
void  pion_xdp_detach(void *h) { (void)h; }
int   pion_xdp_build_tcp_response(void *f, const void *s, int sl, uint8_t fl,
                                   uint32_t sq, uint32_t ak, const void *p, int pl) {
    (void)f;(void)s;(void)sl;(void)fl;(void)sq;(void)ak;(void)p;(void)pl; return -1;
}
int   pion_xdp_extract_tcp_payload(const void *f, int fl, const uint8_t **p,
                                    uint32_t *sq, uint32_t *ak, uint8_t *flags,
                                    uint16_t *sp, uint16_t *dp) {
    (void)f;(void)fl;(void)p;(void)sq;(void)ak;(void)flags;(void)sp;(void)dp; return -1;
}
int   pion_xdp_send_fragmented(void *h, const void *s, int sl, uint8_t fl,
                                uint32_t sq, uint32_t ak, const void *p, int pl) {
    (void)h;(void)s;(void)sl;(void)fl;(void)sq;(void)ak;(void)p;(void)pl; return -1;
}
int   pion_xdp_create_shared_xskmap(int m) { (void)m; return -1; }
int   pion_xdp_load_shared_bpf(uint16_t p, int x) { (void)p;(void)x; return -1; }
int   pion_xdp_attach_bpf(const char *n, int f) { (void)n;(void)f; return -1; }
void  pion_xdp_detach_by_name(const char *n) { (void)n; }
void *pion_xdp_create_worker(const char *n, int q, uint16_t p, int x, int b) {
    (void)n;(void)q;(void)p;(void)x;(void)b; return NULL;
}
void  pion_xdp_destroy_worker(void *h) { (void)h; }
int   pion_xdp_setup_flow_steering(const char *n, uint16_t p, int q) { (void)n;(void)p;(void)q; return -1; }
void  pion_xdp_remove_flow_steering(const char *n, uint16_t p) { (void)n;(void)p; }
void  pion_xdp_register_signal_handlers(const char *n, uint16_t p) { (void)n;(void)p; }

#endif /* __linux__ */
