// ============================================================================
// DAC (Dynamic Airy Chaotic) 5-bit Substitution Box - Verilog HDL
// ============================================================================
// Based on: "IoT enabled data protection with substitution box for
//            lightweight ciphers" - K.B. Sarmila, S.V. Manisekaran
// Egyptian Informatics Journal 29 (2025) 100620
//
// Contents (all combinational): dac_sbox (paper Table 3 as a case lookup),
// dac_sbox_inv, the 32-bit substitution layer (six S-boxes on bits 31..2,
// bits 1 and 0 complemented; paper Fig. 3), the PRESENT-style permutation
// layer and one SPN round.  The tables match src.sbox.DAC_SBOX/DAC_SBOX_INV,
// the permutation matches src.block_cipher.PERM_TABLE, and both facts plus
// every vector of the self-checking testbench dac_sbox_tb.v are verified
// against the Python model by tests/test_hdl.py.
//
// Area / timing: this file states no gate-equivalent or delay figure; such
// numbers depend on the cell library and the flow and are not derived from
// the HDL.  Common-condition synthesis results for the DAC table (Yosys/ABC
// on NanGate45 and SKY130) are produced by run_hardware.py and recorded in
// output/results_hardware.json.  (The base paper reports hardware figures
// for a Cyclone IV FPGA target; they refer to the circuit of that work.)
// ============================================================================

module dac_sbox (
    input  wire [4:0] data_in,   // 5-bit input
    output reg  [4:0] data_out   // 5-bit substituted output
);

    // S-Box Lookup Table (Table 3 from paper)
    // Combinational logic implementation
    always @(*) begin
        case (data_in)
            5'd0:  data_out = 5'd30;
            5'd1:  data_out = 5'd24;
            5'd2:  data_out = 5'd9;
            5'd3:  data_out = 5'd27;
            5'd4:  data_out = 5'd28;
            5'd5:  data_out = 5'd26;
            5'd6:  data_out = 5'd4;
            5'd7:  data_out = 5'd22;
            5'd8:  data_out = 5'd2;
            5'd9:  data_out = 5'd15;
            5'd10: data_out = 5'd29;
            5'd11: data_out = 5'd5;
            5'd12: data_out = 5'd13;
            5'd13: data_out = 5'd8;
            5'd14: data_out = 5'd25;
            5'd15: data_out = 5'd3;
            5'd16: data_out = 5'd11;
            5'd17: data_out = 5'd10;
            5'd18: data_out = 5'd16;
            5'd19: data_out = 5'd31;
            5'd20: data_out = 5'd0;
            5'd21: data_out = 5'd20;
            5'd22: data_out = 5'd14;
            5'd23: data_out = 5'd21;
            5'd24: data_out = 5'd1;
            5'd25: data_out = 5'd12;
            5'd26: data_out = 5'd18;
            5'd27: data_out = 5'd6;
            5'd28: data_out = 5'd23;
            5'd29: data_out = 5'd19;
            5'd30: data_out = 5'd17;
            5'd31: data_out = 5'd7;
            default: data_out = 5'd0;
        endcase
    end

endmodule


// ============================================================================
// Inverse DAC S-Box (for decryption)
// ============================================================================
module dac_sbox_inv (
    input  wire [4:0] data_in,
    output reg  [4:0] data_out
);

    always @(*) begin
        case (data_in)
            5'd0:  data_out = 5'd20;
            5'd1:  data_out = 5'd24;
            5'd2:  data_out = 5'd8;
            5'd3:  data_out = 5'd15;
            5'd4:  data_out = 5'd6;
            5'd5:  data_out = 5'd11;
            5'd6:  data_out = 5'd27;
            5'd7:  data_out = 5'd31;
            5'd8:  data_out = 5'd13;
            5'd9:  data_out = 5'd2;
            5'd10: data_out = 5'd17;
            5'd11: data_out = 5'd16;
            5'd12: data_out = 5'd25;
            5'd13: data_out = 5'd12;
            5'd14: data_out = 5'd22;
            5'd15: data_out = 5'd9;
            5'd16: data_out = 5'd18;
            5'd17: data_out = 5'd30;
            5'd18: data_out = 5'd26;
            5'd19: data_out = 5'd29;
            5'd20: data_out = 5'd21;
            5'd21: data_out = 5'd23;
            5'd22: data_out = 5'd7;
            5'd23: data_out = 5'd28;
            5'd24: data_out = 5'd1;
            5'd25: data_out = 5'd14;
            5'd26: data_out = 5'd5;
            5'd27: data_out = 5'd3;
            5'd28: data_out = 5'd4;
            5'd29: data_out = 5'd10;
            5'd30: data_out = 5'd0;
            5'd31: data_out = 5'd19;
            default: data_out = 5'd0;
        endcase
    end

endmodule


// ============================================================================
// 32-bit Substitution Layer using 6x DAC S-Box + 2-bit remainder handling
// ============================================================================
module dac_substitution_layer (
    input  wire [31:0] data_in,
    output wire [31:0] data_out
);

    // 6 parallel S-Box instances for 5-bit chunks (30 bits)
    wire [4:0] sbox_out_0, sbox_out_1, sbox_out_2;
    wire [4:0] sbox_out_3, sbox_out_4, sbox_out_5;

    dac_sbox sbox0 (.data_in(data_in[31:27]), .data_out(sbox_out_0));
    dac_sbox sbox1 (.data_in(data_in[26:22]), .data_out(sbox_out_1));
    dac_sbox sbox2 (.data_in(data_in[21:17]), .data_out(sbox_out_2));
    dac_sbox sbox3 (.data_in(data_in[16:12]), .data_out(sbox_out_3));
    dac_sbox sbox4 (.data_in(data_in[11:7]),  .data_out(sbox_out_4));
    dac_sbox sbox5 (.data_in(data_in[6:2]),   .data_out(sbox_out_5));

    // 2 remainder bits: flip MSB and LSB
    wire [1:0] remainder = {~data_in[1], ~data_in[0]};

    assign data_out = {sbox_out_0, sbox_out_1, sbox_out_2,
                       sbox_out_3, sbox_out_4, sbox_out_5, remainder};

endmodule


// ============================================================================
// 32-bit Permutation Layer (PRESENT-inspired)
// ============================================================================
module dac_permutation_layer (
    input  wire [31:0] data_in,
    output wire [31:0] data_out
);

    // Permutation table (bit positions)
    assign data_out[31] = data_in[31]; // bit  0 -> pos  0
    assign data_out[25] = data_in[30]; // bit  1 -> pos  6
    assign data_out[19] = data_in[29]; // bit  2 -> pos 12
    assign data_out[13] = data_in[28]; // bit  3 -> pos 18
    assign data_out[7]  = data_in[27]; // bit  4 -> pos 24
    assign data_out[1]  = data_in[26]; // bit  5 -> pos 30
    assign data_out[27] = data_in[25]; // bit  6 -> pos  4
    assign data_out[21] = data_in[24]; // bit  7 -> pos 10
    assign data_out[15] = data_in[23]; // bit  8 -> pos 16
    assign data_out[9]  = data_in[22]; // bit  9 -> pos 22
    assign data_out[3]  = data_in[21]; // bit 10 -> pos 28
    assign data_out[29] = data_in[20]; // bit 11 -> pos  2
    assign data_out[23] = data_in[19]; // bit 12 -> pos  8
    assign data_out[17] = data_in[18]; // bit 13 -> pos 14
    assign data_out[11] = data_in[17]; // bit 14 -> pos 20
    assign data_out[5]  = data_in[16]; // bit 15 -> pos 26
    assign data_out[30] = data_in[15]; // bit 16 -> pos  1
    assign data_out[24] = data_in[14]; // bit 17 -> pos  7
    assign data_out[18] = data_in[13]; // bit 18 -> pos 13
    assign data_out[12] = data_in[12]; // bit 19 -> pos 19
    assign data_out[6]  = data_in[11]; // bit 20 -> pos 25
    assign data_out[0]  = data_in[10]; // bit 21 -> pos 31
    assign data_out[26] = data_in[9];  // bit 22 -> pos  5
    assign data_out[20] = data_in[8];  // bit 23 -> pos 11
    assign data_out[14] = data_in[7];  // bit 24 -> pos 17
    assign data_out[8]  = data_in[6];  // bit 25 -> pos 23
    assign data_out[2]  = data_in[5];  // bit 26 -> pos 29
    assign data_out[28] = data_in[4];  // bit 27 -> pos  3
    assign data_out[22] = data_in[3];  // bit 28 -> pos  9
    assign data_out[16] = data_in[2];  // bit 29 -> pos 15
    assign data_out[10] = data_in[1];  // bit 30 -> pos 21
    assign data_out[4]  = data_in[0];  // bit 31 -> pos 27

endmodule


// ============================================================================
// Single Encryption Round
// ============================================================================
module dac_round (
    input  wire [31:0] data_in,
    input  wire [31:0] round_key,
    input  wire        skip_perm,  // 1 = skip permutation (last round)
    output wire [31:0] data_out
);

    wire [31:0] after_key_mix;
    wire [31:0] after_sub;
    wire [31:0] after_perm;

    // Step 1: Key mixing (XOR)
    assign after_key_mix = data_in ^ round_key;

    // Step 2: Substitution
    dac_substitution_layer sub_layer (
        .data_in(after_key_mix),
        .data_out(after_sub)
    );

    // Step 3: Permutation (conditional)
    dac_permutation_layer perm_layer (
        .data_in(after_sub),
        .data_out(after_perm)
    );

    assign data_out = skip_perm ? after_sub : after_perm;

endmodule
