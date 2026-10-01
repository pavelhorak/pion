/* xdp_kern.c — BPF/XDP program for Pion kernel bypass
 *
 * This is an XDP program compiled to BPF bytecode that runs inside the Linux kernel.
 * It intercepts TCP packets destined for Pion's port and redirects them to an AF_XDP
 * socket for zero-copy userspace processing, bypassing the entire kernel TCP/IP stack.
 *
 * Compilation (requires clang with BPF target):
 *   clang -O2 -target bpf -c src/ffi/xdp_kern.c -o src/ffi/xdp_kern.o
 *
 * The compiled BPF object is loaded at runtime by xdp_wrap.c via libbpf or raw bpf() syscall.
 *
 * Architecture:
 *   NIC driver → XDP hook (this program) → AF_XDP socket → Pion userspace
 *                    ↓ (non-matching traffic)
 *              Linux network stack (normal path)
 */

/* BPF helper definitions — we define these manually to avoid kernel header dependency.
 * These match the Linux kernel's BPF helper IDs (include/uapi/linux/bpf.h). */

typedef unsigned char      __u8;
typedef unsigned short     __u16;
typedef unsigned int       __u32;
typedef unsigned long long __u64;
typedef int                __s32;

/* BPF map types */
#define BPF_MAP_TYPE_XSKMAP 17

/* XDP return codes */
#define XDP_ABORTED  0
#define XDP_DROP     1
#define XDP_PASS     2
#define XDP_TX       3
#define XDP_REDIRECT 4

/* Ethernet header (14 bytes) */
struct ethhdr {
    __u8  h_dest[6];
    __u8  h_source[6];
    __u16 h_proto;
};

/* IPv4 header (20 bytes minimum) */
struct iphdr {
    __u8  ihl_version;    /* version:4, ihl:4 */
    __u8  tos;
    __u16 tot_len;
    __u16 id;
    __u16 frag_off;
    __u8  ttl;
    __u8  protocol;
    __u16 check;
    __u32 saddr;
    __u32 daddr;
};

/* TCP header (20 bytes minimum) */
struct tcphdr {
    __u16 source;
    __u16 dest;
    __u32 seq;
    __u32 ack_seq;
    __u16 flags;         /* data offset:4, reserved:3, flags:9 */
    __u16 window;
    __u16 check;
    __u16 urg_ptr;
};

/* ETH_P_IP in network byte order */
#define ETH_P_IP_BE 0x0008
/* IPPROTO_TCP */
#define IPPROTO_TCP 6

/* BPF context for XDP programs */
struct xdp_md {
    __u32 data;
    __u32 data_end;
    __u32 data_meta;
    __u32 ingress_ifindex;
    __u32 rx_queue_index;
    __u32 egress_ifindex;
};

/* BPF helper: redirect to AF_XDP socket from XSKMAP */
static long (*bpf_redirect_map)(void *map, __u32 key, __u64 flags)
    = (void *)51;

/* BPF map definition — XSKMAP for AF_XDP socket redirection.
 * Each entry maps rx_queue_index → AF_XDP socket fd.
 * Populated by userspace (xdp_wrap.c) after socket creation. */
struct {
    __u32 type;        /* BPF_MAP_TYPE_XSKMAP */
    __u32 key_size;    /* sizeof(__u32) = 4 */
    __u32 value_size;  /* sizeof(__u32) = 4 */
    __u32 max_entries; /* max rx queues */
} xsk_map __attribute__((section("maps"), used)) = {
    .type        = BPF_MAP_TYPE_XSKMAP,
    .key_size    = sizeof(__u32),
    .value_size  = sizeof(__u32),
    .max_entries = 64,
};

/* Global: target port in network byte order.
 * Set by userspace via BPF global variable rewrite before loading.
 * Default: 1974 (0x07B6) → network byte order 0xB607. */
volatile __u32 pion_target_port = 0xB607;

/* XDP program entry point.
 * For every incoming packet:
 *   1. Parse Ethernet → IPv4 → TCP headers
 *   2. If TCP dest port matches pion_target_port → redirect to AF_XDP (zero-copy to userspace)
 *   3. If TCP source port matches (response path) → redirect to AF_XDP
 *   4. Otherwise → XDP_PASS (let kernel handle normally)
 */
__attribute__((section("xdp"), used))
int pion_xdp_filter(struct xdp_md *ctx)
{
    void *data     = (void *)(__u64)ctx->data;
    void *data_end = (void *)(__u64)ctx->data_end;

    /* Parse Ethernet header */
    struct ethhdr *eth = data;
    if ((void *)(eth + 1) > data_end)
        return XDP_PASS;

    /* Only process IPv4 */
    if (eth->h_proto != ETH_P_IP_BE)
        return XDP_PASS;

    /* Parse IPv4 header */
    struct iphdr *ip = (void *)(eth + 1);
    if ((void *)(ip + 1) > data_end)
        return XDP_PASS;

    /* Only process TCP */
    if (ip->protocol != IPPROTO_TCP)
        return XDP_PASS;

    /* Variable-length IP header: ihl is lower 4 bits of ihl_version */
    __u32 ip_hlen = (ip->ihl_version & 0x0F) * 4;
    if (ip_hlen < 20)
        return XDP_PASS;

    /* Parse TCP header */
    struct tcphdr *tcp = (void *)(((__u8 *)ip) + ip_hlen);
    if ((void *)(tcp + 1) > data_end)
        return XDP_PASS;

    /* Check if this packet is for/from Pion's port */
    __u16 target = (__u16)pion_target_port;
    if (tcp->dest != target && tcp->source != target)
        return XDP_PASS;

    /* Redirect to AF_XDP socket via XSKMAP.
     * Key = rx_queue_index — maps to the AF_XDP socket bound to that queue. */
    return bpf_redirect_map(&xsk_map, ctx->rx_queue_index, 0);
}

/* License string required for BPF programs that use certain helpers */
char _license[] __attribute__((section("license"), used)) = "Dual BSD/GPL";
