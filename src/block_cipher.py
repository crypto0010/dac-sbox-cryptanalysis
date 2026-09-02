"""Lightweight SPN Block Cipher using the DAC S-Box.

A complete substitution-permutation network (SPN) that uses the 5-bit DAC
substitution box (or any other bijective 32-entry table) as its only
nonlinear component.

Block size: 32 bits, key size: 80 bits, 16 rounds. Round ``r`` (0..15)::

    state ^= round_key[r]
    state  = S-layer(state)
    state  = P-layer(state)          # skipped in the last round

followed by a final whitening ``state ^= round_key[16]``.

S-layer (paper Section 4.1 / Fig. 3): six aligned 5-bit S-boxes act on
bit positions 31..2 and the two low bits (positions 1 and 0) are merely
complemented. Those two state bits therefore BYPASS the nonlinear layer in
every round; they are only mixed with the rest of the state by the P-layer.
The optional ``full_sbox_layer=True`` mode (not in the paper) adds a seventh
S-box on the wrap-around window (bits 1, 0, 31, 30, 29) so that every state
bit is substituted each round; see :meth:`DACSBox.substitute_block32_full`.

P-layer: the fixed 32-bit bit permutation ``PERM_TABLE`` (PRESENT-style),
applied through four precomputed 256-entry byte tables.

Key schedule (PRESENT-inspired): the 80-bit key register is rotated left by
13 bits, its top 5 bits pass through the S-box and the round counter is XORed
into bits 19..15; the 32-bit round key is the top of the register.

:meth:`DACBlockCipher.encrypt_bytes` / :meth:`~DACBlockCipher.decrypt_bytes`
are raw ECB mode with PKCS#7 padding and no authentication. They exist to
demonstrate the S-box inside a cipher and must not be used to protect data.
"""

from __future__ import annotations

from typing import Any, Dict

from collections.abc import Sequence

import numpy as np

from .sbox import DACSBox, invert_permutation


#: Single-block demonstration vector shared by :func:`demo_cipher` and
#: ``main.run_cipher_demo`` so the printed and the recorded vector cannot drift.
DEMO_PLAINTEXT = 0xDEADBEEF

BLOCK_SIZE = 32  # bits
KEY_SIZE = 80    # bits
NUM_ROUNDS = 16
BLOCK_BYTES = BLOCK_SIZE // 8

# Bit permutation: the state bit at MSB-first index i (bit 31 - i) moves to
# MSB-first index PERM_TABLE[i]. Inspired by the PRESENT cipher.
PERM_TABLE: tuple[int, ...] = (
     0,  6, 12, 18, 24, 30,  4, 10,
    16, 22, 28,  2,  8, 14, 20, 26,
     1,  7, 13, 19, 25, 31,  5, 11,
    17, 23, 29,  3,  9, 15, 21, 27,
)
INV_PERM_TABLE: tuple[int, ...] = tuple(invert_permutation(PERM_TABLE))


def permute_bits(state: int, perm: Sequence[int] = PERM_TABLE) -> int:
    """Reference bit permutation of a 32-bit ``state`` (one loop iteration per bit).

    The bit at MSB-first index ``i`` is moved to MSB-first index ``perm[i]``.
    """
    result = 0
    for i, j in enumerate(perm):
        result |= ((state >> (BLOCK_SIZE - 1 - i)) & 1) << (BLOCK_SIZE - 1 - j)
    return result


def byte_permutation_tables(perm: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    """Precompute the bit permutation ``perm`` as four 256-entry byte tables.

    ``tables[k][b]`` is the permuted image of byte value ``b`` placed in byte
    ``k`` of the block (``k = 0`` is the most significant byte). Because a bit
    permutation is linear, the image of a block is the OR of its four byte
    images, which is what :func:`permute_block32` evaluates.
    """
    return tuple(
        tuple(permute_bits(b << (BLOCK_SIZE - 8 * (k + 1)), perm) for b in range(256))
        for k in range(BLOCK_BYTES))


PERM_BYTE_TABLES = byte_permutation_tables(PERM_TABLE)
INV_PERM_BYTE_TABLES = byte_permutation_tables(INV_PERM_TABLE)


def permute_block32(state: int, tables: tuple[tuple[int, ...], ...] = PERM_BYTE_TABLES) -> int:
    """Apply a bit permutation given as byte tables (see :func:`byte_permutation_tables`).

    No input validation: the caller guarantees ``0 <= state < 2**32``.
    """
    t0, t1, t2, t3 = tables
    return (t0[state >> 24] | t1[(state >> 16) & 0xFF]
            | t2[(state >> 8) & 0xFF] | t3[state & 0xFF])


class DACBlockCipher:
    """32-bit / 80-bit / 16-round SPN cipher built on a 5-bit S-box.

    Args:
        key: 80-bit master key as a non-negative integer.
        sbox_table: 32-entry bijective table or a :class:`DACSBox`; the
            paper's DAC S-box when ``None``.
        full_sbox_layer: if True every state bit passes through the S-box each
            round (seventh wrap-around S-box, see module docstring). The default
            ``False`` is the paper's layout, in which the two low state bits
            bypass the S-layer and are only complemented.

    Attributes:
        key, sbox, full_sbox_layer: as passed (``sbox`` is a :class:`DACSBox`).
        round_keys: tuple of ``NUM_ROUNDS + 1`` 32-bit round keys.

    Raises:
        TypeError: if ``key`` is not an integer.
        ValueError: if ``key`` is not an 80-bit unsigned integer, or the S-box
            table is not a 32-entry permutation.
    """

    BLOCK_SIZE = BLOCK_SIZE
    KEY_SIZE = KEY_SIZE
    NUM_ROUNDS = NUM_ROUNDS
    PERM_TABLE = PERM_TABLE
    INV_PERM_TABLE = INV_PERM_TABLE

    def __init__(self, key: int,
                 sbox_table: Sequence[int] | np.ndarray | DACSBox | None = None,
                 full_sbox_layer: bool = False) -> None:
        if isinstance(key, bool) or not isinstance(key, (int, np.integer)):
            raise TypeError(f"key must be an integer, got {type(key).__name__}")
        key = int(key)
        if not 0 <= key < (1 << KEY_SIZE):
            raise ValueError(f"key must be a {KEY_SIZE}-bit unsigned integer")
        self.key = key
        self.sbox = sbox_table if isinstance(sbox_table, DACSBox) else DACSBox(sbox_table)
        self.full_sbox_layer = bool(full_sbox_layer)
        if self.full_sbox_layer:
            self._substitute = self.sbox.substitute_block32_full
            self._inverse_substitute = self.sbox.inverse_substitute_block32_full
        else:
            self._substitute = self.sbox.substitute_block32
            self._inverse_substitute = self.sbox.inverse_substitute_block32
        self.round_keys: tuple[int, ...] = self._generate_round_keys()

    def _generate_round_keys(self) -> tuple[int, ...]:
        """Derive ``NUM_ROUNDS + 1`` round keys from the master key (see module docstring)."""
        key_mask = (1 << KEY_SIZE) - 1
        low_mask = (1 << (KEY_SIZE - 5)) - 1
        table = self.sbox.table
        round_keys = []
        register = self.key
        for round_num in range(NUM_ROUNDS + 1):
            round_keys.append(register >> (KEY_SIZE - BLOCK_SIZE))
            # Rotate left by 13, substitute the top 5 bits, XOR the round counter.
            register = ((register << 13) | (register >> (KEY_SIZE - 13))) & key_mask
            register = (register & low_mask) | (table[register >> (KEY_SIZE - 5)] << (KEY_SIZE - 5))
            register ^= ((round_num + 1) & 0x1F) << 15
        return tuple(round_keys)

    @staticmethod
    def _check_block(value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"block must be an integer, got {type(value).__name__}")
        if not 0 <= value < (1 << BLOCK_SIZE):
            raise ValueError(f"block must be a {BLOCK_SIZE}-bit unsigned integer, got {value}")

    def substitution_layer(self, state: int) -> int:
        """S-layer of one round (no input validation)."""
        return self._substitute(state)

    def inverse_substitution_layer(self, state: int) -> int:
        """Inverse S-layer (no input validation)."""
        return self._inverse_substitute(state)

    @staticmethod
    def permutation_layer(state: int) -> int:
        """P-layer: bit permutation ``PERM_TABLE`` (no input validation)."""
        return permute_block32(state, PERM_BYTE_TABLES)

    @staticmethod
    def inverse_permutation_layer(state: int) -> int:
        """Inverse P-layer (no input validation)."""
        return permute_block32(state, INV_PERM_BYTE_TABLES)

    def encrypt_block(self, plaintext: int) -> int:
        """Encrypt one 32-bit block given as an integer.

        Raises:
            TypeError, ValueError: if ``plaintext`` is not a 32-bit unsigned integer.
        """
        self._check_block(plaintext)
        round_keys = self.round_keys
        substitute = self._substitute
        permute = permute_block32
        state = int(plaintext)
        for r in range(NUM_ROUNDS - 1):
            state = permute(substitute(state ^ round_keys[r]))
        state = substitute(state ^ round_keys[NUM_ROUNDS - 1])
        return state ^ round_keys[NUM_ROUNDS]

    def decrypt_block(self, ciphertext: int) -> int:
        """Decrypt one 32-bit block given as an integer.

        Raises:
            TypeError, ValueError: if ``ciphertext`` is not a 32-bit unsigned integer.
        """
        self._check_block(ciphertext)
        round_keys = self.round_keys
        inverse_substitute = self._inverse_substitute
        permute = permute_block32
        inv_tables = INV_PERM_BYTE_TABLES
        state = int(ciphertext) ^ round_keys[NUM_ROUNDS]
        state = inverse_substitute(state) ^ round_keys[NUM_ROUNDS - 1]
        for r in range(NUM_ROUNDS - 2, -1, -1):
            state = inverse_substitute(permute(state, inv_tables)) ^ round_keys[r]
        return state

    def encrypt_bytes(self, data: bytes | bytearray | memoryview) -> bytes:
        """Encrypt a byte string in raw ECB mode with PKCS#7 padding (demonstration only).

        The input is padded to a multiple of 4 bytes (1..4 padding bytes, each
        equal to the padding length) and every 4-byte block is encrypted
        independently. ECB leaks equal-block patterns and provides no
        integrity protection, so this must not be used for real data.

        Raises:
            ValueError: if ``data`` is empty.
        """
        data = bytes(data)
        if not data:
            raise ValueError("encrypt_bytes: input must not be empty")
        pad_len = BLOCK_BYTES - len(data) % BLOCK_BYTES
        padded = data + bytes([pad_len]) * pad_len
        encrypt = self.encrypt_block
        return b"".join(
            encrypt(int.from_bytes(padded[i:i + BLOCK_BYTES], "big")).to_bytes(BLOCK_BYTES, "big")
            for i in range(0, len(padded), BLOCK_BYTES))

    def decrypt_bytes(self, data: bytes | bytearray | memoryview) -> bytes:
        """Decrypt the output of :meth:`encrypt_bytes` and strip the PKCS#7 padding.

        Raises:
            ValueError: if ``data`` is empty, its length is not a multiple of
                4, or the decrypted padding is invalid (wrong key, corrupted or
                truncated ciphertext).
        """
        data = bytes(data)
        if not data:
            raise ValueError("decrypt_bytes: input must not be empty")
        if len(data) % BLOCK_BYTES:
            raise ValueError(f"decrypt_bytes: ciphertext length must be a multiple of "
                             f"{BLOCK_BYTES} bytes, got {len(data)}")
        decrypt = self.decrypt_block
        plain = b"".join(
            decrypt(int.from_bytes(data[i:i + BLOCK_BYTES], "big")).to_bytes(BLOCK_BYTES, "big")
            for i in range(0, len(data), BLOCK_BYTES))
        pad_len = plain[-1]
        if not 1 <= pad_len <= BLOCK_BYTES or plain[-pad_len:] != bytes([pad_len]) * pad_len:
            raise ValueError("decrypt_bytes: invalid PKCS#7 padding "
                             "(wrong key or corrupted ciphertext)")
        return plain[:-pad_len]


def demo_cipher() -> Dict[str, Any]:
    """Demonstrate the DAC block cipher (single block, byte string, avalanche).

    Returns every value it computes and prints -- including the cipher itself
    under ``"cipher"`` -- so a caller can record the demonstration without
    evaluating the same test vector a second time.
    """
    print("\n" + "=" * 60)
    print("  DAC LIGHTWEIGHT BLOCK CIPHER DEMO")
    print("=" * 60)

    key = 0x0123456789ABCDEF0123
    cipher = DACBlockCipher(key)

    plaintext = DEMO_PLAINTEXT
    print(f"\n  Key:        0x{key:020X}")
    print(f"  Plaintext:  0x{plaintext:08X}")
    ciphertext = cipher.encrypt_block(plaintext)
    print(f"  Ciphertext: 0x{ciphertext:08X}")
    decrypted = cipher.decrypt_block(ciphertext)
    print(f"  Decrypted:  0x{decrypted:08X}")
    print(f"  Correct:    {decrypted == plaintext}")

    message = b"Hello, IoT!"
    print(f"\n  Message: {message}  (raw ECB + PKCS#7, demonstration only)")
    encrypted = cipher.encrypt_bytes(message)
    print(f"  Encrypted: {encrypted.hex()}")
    decrypted_msg = cipher.decrypt_bytes(encrypted)
    print(f"  Decrypted: {decrypted_msg}")
    print(f"  Correct:   {decrypted_msg == message}")

    print("\n  AVALANCHE TEST (single bit flip in plaintext):")
    pt1 = plaintext
    pt2 = pt1 ^ 1  # Flip LSB
    ct1 = cipher.encrypt_block(pt1)
    ct2 = cipher.encrypt_block(pt2)
    diff_bits = (ct1 ^ ct2).bit_count()
    print(f"  PT1: 0x{pt1:08X} -> CT1: 0x{ct1:08X}")
    print(f"  PT2: 0x{pt2:08X} -> CT2: 0x{ct2:08X}")
    print(f"  Bits changed in ciphertext: {diff_bits}/{BLOCK_SIZE} "
          f"({diff_bits / BLOCK_SIZE * 100:.1f}%)")
    print("\n  Note: bits 1 and 0 of the state bypass the S-layer every round "
          "(paper layout); use full_sbox_layer=True to substitute all 32 bits.")

    return {
        "cipher": cipher,
        "key": f"0x{key:020X}",
        "plaintext": f"0x{plaintext:08X}",
        "ciphertext": f"0x{ciphertext:08X}",
        "decrypted": f"0x{decrypted:08X}",
        "round_trip_ok": decrypted == plaintext,
        "message": message.decode("ascii"),
        "encrypted_hex": encrypted.hex(),
        "decrypted_message": decrypted_msg.decode("ascii"),
        "message_round_trip_ok": decrypted_msg == message,
        "avalanche_plaintext_pair": [f"0x{pt1:08X}", f"0x{pt2:08X}"],
        "avalanche_bits_lsb_flip": diff_bits,
        "block_bits": BLOCK_SIZE,
    }
