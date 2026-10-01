# Mojo compiler repros

Minimal programs for Mojo 1.0.0 defects that Pion has had to code around.
Each file's header states the command, the expected output and what happens instead.
They are not tests; nothing builds them automatically.

| file | defect | Pion's guard |
|---|---|---|
| `tail_call_drops_stack_stores.mojo` | An out-of-line call whose pointer argument has an untracked origin is emitted as LLVM `tail call`, so -O3 deletes stores into a caller stack buffer the callee reads (and loses the callee's writes into one). | `tests/test_audit_tail_alloca.py` (gate tier; also `pixi run audit-tail-alloca`) must report 0 |
| `short_string_ptr_tail_call.mojo` | Same defect, reached through a short String's inline storage (9–23 bytes). | same |
| `cast_chain_sign_extends.mojo` | `UInt64(int8_ptr[i].cast[DType.uint8]())` sign-extends. | `_byte_u64` in `src/vector/kernels.mojo` |

The rule that keeps Pion safe: a pointer into a stack local (`stack_allocation`,
an `InlineArray`, a short `String`'s bytes) may cross only into an
`@always_inline` Mojo function or into C through `external_call`. A heap
pointer can go anywhere.
