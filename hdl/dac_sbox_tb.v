`timescale 1ns / 1ps
// ============================================================================
// Self-checking testbench for hdl/dac_sbox.v
// ============================================================================
// Every expected value below is derived from the Python reference model
// (src.sbox.DACSBox, src.block_cipher) and is re-derived from it by
// tests/test_hdl.py, so the vectors cannot drift away from the software.
//
// Checks
//   1. dac_sbox              : all 32 inputs against Table 3
//   2. dac_sbox_inv          : all 32 inputs against the inverse table, and
//                              S^-1(S(x)) == x through a chained instance
//   3. bijectivity           : the 32 forward outputs are pairwise distinct
//   4. dac_permutation_layer : every single-bit input (the layer is linear,
//                              so these cover it completely) plus dense vectors
//   5. dac_round             : (data, round key, skip_perm) vectors; with a
//                              zero key and skip_perm = 1 the round is the
//                              bare substitution layer, including the two
//                              complemented low bits
//
// Prints a PASS/FAIL summary; any mismatch ends the simulation with $fatal,
// i.e. a non-zero simulator exit status.  Run with, for example:
//   iverilog -g2012 -o dac_sbox_sim hdl/dac_sbox.v hdl/dac_sbox_tb.v
//   vvp dac_sbox_sim
// ============================================================================
module dac_sbox_tb;

    localparam N_PERM  = 6;   // dense permutation vectors
    localparam N_ROUND = 8;   // round vectors

    reg  [4:0]  x;
    wire [4:0]  s_x;          // dac_sbox(x)
    wire [4:0]  inv_x;        // dac_sbox_inv(x)
    wire [4:0]  round_trip;   // dac_sbox_inv(dac_sbox(x))
    reg  [31:0] perm_in;
    wire [31:0] perm_out;
    reg  [31:0] round_in;
    reg  [31:0] round_key;
    reg         round_skip;
    wire [31:0] round_out;

    reg [4:0]  expected_sbox  [0:31];        // Table 3 (src.sbox.DAC_SBOX)
    reg [4:0]  expected_inv   [0:31];        // src.sbox.DAC_SBOX_INV
    reg [4:0]  perm_pos       [0:31];        // src.block_cipher.PERM_TABLE
    reg [31:0] perm_vec_in    [0:N_PERM-1];  // dense inputs
    reg [31:0] perm_vec_out   [0:N_PERM-1];  // src.block_cipher.permute_bits
    reg [31:0] round_vec_in   [0:N_ROUND-1];
    reg [31:0] round_vec_key  [0:N_ROUND-1];
    reg        round_vec_skip [0:N_ROUND-1];
    reg [31:0] round_vec_out  [0:N_ROUND-1]; // P(S(in ^ key)), P skipped per flag

    reg [31:0] seen;
    reg [31:0] expected_bit;
    integer i;
    integer errors_sbox, errors_inv, errors_round_trip, errors_bijective;
    integer errors_perm, errors_round, errors;

    dac_sbox              dut_sbox  (.data_in(x),   .data_out(s_x));
    dac_sbox_inv          dut_inv   (.data_in(x),   .data_out(inv_x));
    dac_sbox_inv          dut_chain (.data_in(s_x), .data_out(round_trip));
    dac_permutation_layer dut_perm  (.data_in(perm_in), .data_out(perm_out));
    dac_round             dut_round (.data_in(round_in), .round_key(round_key),
                                     .skip_perm(round_skip), .data_out(round_out));

    initial begin
        // S(x), x = 0..31 (paper Table 3)
        expected_sbox[0]  = 5'd30; expected_sbox[1]  = 5'd24; expected_sbox[2]  = 5'd9;  expected_sbox[3]  = 5'd27;
        expected_sbox[4]  = 5'd28; expected_sbox[5]  = 5'd26; expected_sbox[6]  = 5'd4;  expected_sbox[7]  = 5'd22;
        expected_sbox[8]  = 5'd2;  expected_sbox[9]  = 5'd15; expected_sbox[10] = 5'd29; expected_sbox[11] = 5'd5;
        expected_sbox[12] = 5'd13; expected_sbox[13] = 5'd8;  expected_sbox[14] = 5'd25; expected_sbox[15] = 5'd3;
        expected_sbox[16] = 5'd11; expected_sbox[17] = 5'd10; expected_sbox[18] = 5'd16; expected_sbox[19] = 5'd31;
        expected_sbox[20] = 5'd0;  expected_sbox[21] = 5'd20; expected_sbox[22] = 5'd14; expected_sbox[23] = 5'd21;
        expected_sbox[24] = 5'd1;  expected_sbox[25] = 5'd12; expected_sbox[26] = 5'd18; expected_sbox[27] = 5'd6;
        expected_sbox[28] = 5'd23; expected_sbox[29] = 5'd19; expected_sbox[30] = 5'd17; expected_sbox[31] = 5'd7;

        // S^-1(x), x = 0..31
        expected_inv[0]  = 5'd20; expected_inv[1]  = 5'd24; expected_inv[2]  = 5'd8;  expected_inv[3]  = 5'd15;
        expected_inv[4]  = 5'd6;  expected_inv[5]  = 5'd11; expected_inv[6]  = 5'd27; expected_inv[7]  = 5'd31;
        expected_inv[8]  = 5'd13; expected_inv[9]  = 5'd2;  expected_inv[10] = 5'd17; expected_inv[11] = 5'd16;
        expected_inv[12] = 5'd25; expected_inv[13] = 5'd12; expected_inv[14] = 5'd22; expected_inv[15] = 5'd9;
        expected_inv[16] = 5'd18; expected_inv[17] = 5'd30; expected_inv[18] = 5'd26; expected_inv[19] = 5'd29;
        expected_inv[20] = 5'd21; expected_inv[21] = 5'd23; expected_inv[22] = 5'd7;  expected_inv[23] = 5'd28;
        expected_inv[24] = 5'd1;  expected_inv[25] = 5'd14; expected_inv[26] = 5'd5;  expected_inv[27] = 5'd3;
        expected_inv[28] = 5'd4;  expected_inv[29] = 5'd10; expected_inv[30] = 5'd0;  expected_inv[31] = 5'd19;

        // PERM_TABLE: the state bit at MSB-first index i moves to index perm_pos[i]
        perm_pos[0]  = 5'd0;  perm_pos[1]  = 5'd6;  perm_pos[2]  = 5'd12; perm_pos[3]  = 5'd18;
        perm_pos[4]  = 5'd24; perm_pos[5]  = 5'd30; perm_pos[6]  = 5'd4;  perm_pos[7]  = 5'd10;
        perm_pos[8]  = 5'd16; perm_pos[9]  = 5'd22; perm_pos[10] = 5'd28; perm_pos[11] = 5'd2;
        perm_pos[12] = 5'd8;  perm_pos[13] = 5'd14; perm_pos[14] = 5'd20; perm_pos[15] = 5'd26;
        perm_pos[16] = 5'd1;  perm_pos[17] = 5'd7;  perm_pos[18] = 5'd13; perm_pos[19] = 5'd19;
        perm_pos[20] = 5'd25; perm_pos[21] = 5'd31; perm_pos[22] = 5'd5;  perm_pos[23] = 5'd11;
        perm_pos[24] = 5'd17; perm_pos[25] = 5'd23; perm_pos[26] = 5'd29; perm_pos[27] = 5'd3;
        perm_pos[28] = 5'd9;  perm_pos[29] = 5'd15; perm_pos[30] = 5'd21; perm_pos[31] = 5'd27;

        // Dense permutation vectors (src.block_cipher.permute_bits)
        perm_vec_in[0] = 32'hFFFFFFFF; perm_vec_out[0] = 32'hFFFFFFFF;
        perm_vec_in[1] = 32'hDEADBEEF; perm_vec_out[1] = 32'hCEC7F5FF;
        perm_vec_in[2] = 32'h12345678; perm_vec_out[2] = 32'h3D42310D;
        perm_vec_in[3] = 32'h80000001; perm_vec_out[3] = 32'h80000010;
        perm_vec_in[4] = 32'hC1651DE4; perm_vec_out[4] = 32'h8233536D;
        perm_vec_in[5] = 32'hAF905095; perm_vec_out[5] = 32'hB929D092;

        // Round vectors: round_out = P(S(in ^ key)), P skipped when skip = 1.
        // Vectors 0-3 (zero key or zero data, skip = 1) exercise the bare
        // substitution layer, including the two complemented low bits.
        round_vec_in[0] = 32'h00000000; round_vec_key[0] = 32'h00000000; round_vec_skip[0] = 1'b1; round_vec_out[0] = 32'hF7BDEF7B;
        round_vec_in[1] = 32'hFFFFFFFF; round_vec_key[1] = 32'h00000000; round_vec_skip[1] = 1'b1; round_vec_out[1] = 32'h39CE739C;
        round_vec_in[2] = 32'hDEADBEEF; round_vec_key[2] = 32'h00000000; round_vec_skip[2] = 1'b1; round_vec_out[2] = 32'h349C6998;
        round_vec_in[3] = 32'h00000000; round_vec_key[3] = 32'hDEADBEEF; round_vec_skip[3] = 1'b1; round_vec_out[3] = 32'h349C6998;
        round_vec_in[4] = 32'h12345678; round_vec_key[4] = 32'h0123ABCD; round_vec_skip[4] = 1'b0; round_vec_out[4] = 32'h0FB41E84;
        round_vec_in[5] = 32'hDEADBEEF; round_vec_key[5] = 32'h01234567; round_vec_skip[5] = 1'b0; round_vec_out[5] = 32'h048F765E;
        round_vec_in[6] = 32'h0C274FE1; round_vec_key[6] = 32'hDD56EDF6; round_vec_skip[6] = 1'b0; round_vec_out[6] = 32'h8C44A967;
        round_vec_in[7] = 32'h249B5DCF; round_vec_key[7] = 32'h64A0BCDA; round_vec_skip[7] = 1'b1; round_vec_out[7] = 32'h17A714EA;

        errors_sbox = 0; errors_inv = 0; errors_round_trip = 0;
        errors_bijective = 0; errors_perm = 0; errors_round = 0;
        seen = 32'd0;
        x = 5'd0; perm_in = 32'd0;
        round_in = 32'd0; round_key = 32'd0; round_skip = 1'b0;

        // --- 1-3: forward table, inverse table, S^-1(S(x)), bijectivity ---
        for (i = 0; i < 32; i = i + 1) begin
            x = i[4:0];
            #10;
            if (s_x !== expected_sbox[i]) begin
                errors_sbox = errors_sbox + 1;
                $display("  FAIL dac_sbox(%0d) = %0d, expected %0d", i, s_x, expected_sbox[i]);
            end
            if (inv_x !== expected_inv[i]) begin
                errors_inv = errors_inv + 1;
                $display("  FAIL dac_sbox_inv(%0d) = %0d, expected %0d", i, inv_x, expected_inv[i]);
            end
            if (round_trip !== x) begin
                errors_round_trip = errors_round_trip + 1;
                $display("  FAIL S^-1(S(%0d)) = %0d", i, round_trip);
            end
            seen = seen | (32'd1 << s_x);
        end
        if (seen !== 32'hFFFFFFFF) begin
            errors_bijective = 1;
            $display("  FAIL dac_sbox outputs are not distinct (mask %h)", seen);
        end

        // --- 4: permutation layer ---
        for (i = 0; i < 32; i = i + 1) begin
            perm_in = 32'd1 << (31 - i);
            expected_bit = 32'd1 << (31 - perm_pos[i]);
            #10;
            if (perm_out !== expected_bit) begin
                errors_perm = errors_perm + 1;
                $display("  FAIL permutation of bit index %0d: %h, expected %h",
                         i, perm_out, expected_bit);
            end
        end
        for (i = 0; i < N_PERM; i = i + 1) begin
            perm_in = perm_vec_in[i];
            #10;
            if (perm_out !== perm_vec_out[i]) begin
                errors_perm = errors_perm + 1;
                $display("  FAIL permutation(%h) = %h, expected %h",
                         perm_vec_in[i], perm_out, perm_vec_out[i]);
            end
        end

        // --- 5: one encryption round ---
        for (i = 0; i < N_ROUND; i = i + 1) begin
            round_in   = round_vec_in[i];
            round_key  = round_vec_key[i];
            round_skip = round_vec_skip[i];
            #10;
            if (round_out !== round_vec_out[i]) begin
                errors_round = errors_round + 1;
                $display("  FAIL round(%h, %h, skip=%b) = %h, expected %h",
                         round_vec_in[i], round_vec_key[i], round_vec_skip[i],
                         round_out, round_vec_out[i]);
            end
        end

        // --- summary ---
        errors = errors_sbox + errors_inv + errors_round_trip
               + errors_bijective + errors_perm + errors_round;
        $display("dac_sbox_tb summary: sbox %0d/32, inverse %0d/32, S^-1(S(x)) %0d/32,",
                 errors_sbox, errors_inv, errors_round_trip);
        $display("                     permutation %0d/%0d, round %0d/%0d mismatches",
                 errors_perm, 32 + N_PERM, errors_round, N_ROUND);
        if (errors_bijective == 0)
            $display("dac_sbox_tb bijectivity: PASS");
        else
            $display("dac_sbox_tb bijectivity: FAIL");
        if (errors != 0) begin
            $display("dac_sbox_tb: FAIL (%0d mismatches)", errors);
            $fatal(1, "dac_sbox_tb failed");
        end
        $display("dac_sbox_tb: PASS");
        $finish;
    end

endmodule
