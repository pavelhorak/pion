# pion_resp: parse pipelined RESP requests the way Pion's server does.
from pion_resp import RESP3Parser, RESP3Token, MAX_CMD_TOKENS, MAX_CMD_ENDS, PROTOCOL_ERROR, PE_MULTIBULK_LEN
from std.memory import alloc
from std.testing import assert_equal, assert_true


struct Parsed:
    var parser: RESP3Parser
    var tokens: Pointer[RESP3Token, MutUntrackedOrigin]
    var cmd_ends: Pointer[Int, MutUntrackedOrigin]
    var cmd_byte_ends: Pointer[Int, MutUntrackedOrigin]
    var buf: Pointer[UInt8, MutUntrackedOrigin]
    var num_tokens: Int
    var consumed: Int
    var num_cmds: Int
    var need: Int

    def __init__(out self, text: String) raises:
        self.parser = RESP3Parser()
        self.tokens = alloc[RESP3Token](MAX_CMD_TOKENS)
        self.cmd_ends = alloc[Int](MAX_CMD_ENDS)
        self.cmd_byte_ends = alloc[Int](MAX_CMD_ENDS)
        var n = text.byte_length()
        self.buf = alloc[UInt8](n + 1)
        var src = text.unsafe_ptr()
        for i in range(n):
            self.buf[i] = src[i]
        self.num_tokens = 0
        self.consumed = 0
        self.num_cmds = 0
        self.need = 0
        self.parser.parse_stream(self.buf, n, self.tokens, self.num_tokens, self.consumed,
                                 self.cmd_ends, self.cmd_byte_ends, self.num_cmds, self.need)

    def token(self, i: Int) -> String:
        return self.tokens[i].value()


def test_pipelined_arrays() raises:
    var req = String("*3\r\n$3\r\nSET\r\n$3\r\nkey\r\n$5\r\nvalue\r\n*1\r\n$4\r\nPING\r\n")
    var p = Parsed(req)
    assert_equal(p.num_cmds, 2)
    assert_equal(p.num_tokens, 4)
    assert_equal(p.token(0), "SET")
    assert_equal(p.token(1), "key")
    assert_equal(p.token(2), "value")
    assert_equal(p.token(3), "PING")
    assert_equal(p.cmd_ends[0], 3)
    assert_equal(p.cmd_ends[1], 4)
    assert_equal(p.consumed, req.byte_length())


def test_partial_frame_waits() raises:
    # The second command is cut mid-bulk: only the first is consumed.
    var first = String("*2\r\n$3\r\nGET\r\n$1\r\nk\r\n")
    var p = Parsed(first + "*2\r\n$3\r\nGET\r\n$4\r\nlo")
    assert_equal(p.num_cmds, 1)
    assert_equal(p.consumed, first.byte_length())
    var q = Parsed(String("*2\r\n$3\r\nGE"))
    assert_equal(q.num_cmds, 0)
    assert_equal(q.consumed, 0)


def test_inline_command() raises:
    var p = Parsed(String("SET greeting \"hello world\"\r\n"))
    assert_equal(p.num_cmds, 1)
    assert_equal(p.num_tokens, 3)
    assert_equal(p.token(2), "hello world")


def test_binary_safe_bulk() raises:
    # A bulk string carries any bytes, CR LF included.
    var p = Parsed(String("*2\r\n$4\r\nECHO\r\n$4\r\na\r\nb\r\n"))
    assert_equal(p.num_cmds, 1)
    assert_equal(p.tokens[1].length, 4)


def test_protocol_error() raises:
    var p = Parsed(String("*abc\r\n"))
    assert_equal(p.num_tokens, PROTOCOL_ERROR)
    assert_equal(p.need, PE_MULTIBULK_LEN)


def main() raises:
    test_pipelined_arrays()
    test_partial_frame_waits()
    test_inline_command()
    test_binary_safe_bulk()
    test_protocol_error()
    print("pion_resp: all tests passed")
