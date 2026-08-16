`timescale 1ns / 1ps

// Population LayerNorm in signed Q16.16.  Gamma, beta, input and output share
// Q16.16; variance is Q32.32 and epsilon is exactly round(1e-5 * 2^32).
module gptneo_layernorm #(
    parameter integer D = 64
) (
    input wire clk,
    input wire rst,
    input wire in_valid,
    output wire in_ready,
    input wire signed [31:0] in_x,
    input wire signed [31:0] in_gamma,
    input wire signed [31:0] in_beta,
    input wire start,
    output reg out_valid,
    input wire out_ready,
    output reg [$clog2(D)-1:0] out_index,
    output reg signed [31:0] out_y,
    output wire busy,
    output wire [95:0] debug_status
);
    localparam [63:0] EPSILON_Q32 = 64'd42950;
    localparam LOAD=4'd0, MEAN=4'd1, VAR=4'd2, SQRT_ALIGN=4'd3,
               SQRT_STEP=4'd4, NORM_START=4'd5, NORM_WAIT=4'd6,
               NORM_AFFINE=4'd7, EMIT=4'd8;
    reg [3:0] state;
    reg signed [31:0] xmem[0:D-1], gmem[0:D-1], bmem[0:D-1];
    reg [$clog2(D+1)-1:0] load_count;
    reg [$clog2(D)-1:0] index;
    reg signed [63:0] sum;
    reg signed [63:0] mean;
    // A difference between two signed 32-bit Q16.16 values needs 33 bits.
    // Keep the square at its mathematical width so synthesis does not build
    // a 64x64 multiplier whose unused upper operand bits dominate the XC7
    // implementation.
    reg [71:0] square_sum;
    reg signed [32:0] delta;
    reg [32:0] delta_magnitude;
    reg [65:0] delta_square;
    reg [63:0] variance;
    reg [63:0] deviation;
    // Population normalization bounds |delta / deviation| below sqrt(D-1).
    // For D=64 its Q16.16 representation needs fewer than 20 signed bits;
    // retain a full signed word, but do not feed sign-extension padding into a
    // needlessly wide DSP cascade.
    reg signed [31:0] normalized;
    reg signed [63:0] affine;
    reg [63:0] sqrt_operand, sqrt_result, sqrt_bit;
    reg signed [31:0] debug_y0;
    reg signed [31:0] debug_normalized0;
    reg signed [31:0] debug_affine0;
    reg debug_capture_active;
    reg debug_capture_done;
    reg norm_div_start;
    reg signed [95:0] norm_div_numerator;
    reg [63:0] norm_div_denominator;
    wire norm_div_busy;
    wire norm_div_done;
    wire signed [31:0] norm_div_quotient;

    gptneo_iterative_divider norm_divider(
        .clk(clk), .rst(rst), .start(norm_div_start),
        .numerator(norm_div_numerator), .denominator(norm_div_denominator),
        .busy(norm_div_busy), .done(norm_div_done),
        .quotient(norm_div_quotient)
    );

    assign in_ready = (state == LOAD) && (load_count < D);
    assign busy = (state != LOAD) || (load_count != 0);
    assign debug_status = {debug_affine0, debug_normalized0, debug_y0};

    always @(posedge clk) begin
        if (rst) begin
            state <= LOAD; load_count <= 0; sum <= 0; out_valid <= 0;
            index <= 0; square_sum <= 0; debug_y0 <= 0;
            debug_normalized0 <= 0; debug_affine0 <= 0;
            debug_capture_active <= 0; debug_capture_done <= 0;
            norm_div_start <= 0; norm_div_numerator <= 0;
            norm_div_denominator <= 1;
        end else begin
            norm_div_start <= 0;
            if (in_valid && in_ready) begin
                if (load_count == 0 && !debug_capture_done) begin
                    debug_capture_active <= 1;
                end
                xmem[load_count] <= in_x;
                gmem[load_count] <= in_gamma;
                bmem[load_count] <= in_beta;
                sum <= sum + in_x;
                load_count <= load_count + 1'b1;
            end
            case (state)
                LOAD: if (start && load_count == D) state <= MEAN;
                MEAN: begin
                    mean <= sum / D;
                    square_sum <= 0;
                    index <= 0;
                    state <= VAR;
                end
                VAR: begin
                    delta = $signed({xmem[index][31], xmem[index]})
                            - $signed(mean[32:0]);
                    delta_magnitude = delta < 0 ? -delta : delta;
                    delta_square = delta_magnitude * delta_magnitude;
                    if (index == D-1) begin
                        variance = (square_sum + delta_square) / D + EPSILON_Q32;
                        sqrt_operand <= variance;
                        sqrt_result <= 0;
                        sqrt_bit <= 64'h4000000000000000;
                        index <= 0;
                        state <= SQRT_ALIGN;
                    end else begin
                        square_sum <= square_sum + delta_square;
                        index <= index + 1'b1;
                    end
                end
                SQRT_ALIGN: begin
                    if (sqrt_bit > sqrt_operand) sqrt_bit <= sqrt_bit >> 2;
                    else state <= SQRT_STEP;
                end
                SQRT_STEP: begin
                    if (sqrt_bit == 0) begin
                        deviation <= sqrt_result;
                        state <= NORM_START;
                    end else begin
                        if (sqrt_operand >= sqrt_result + sqrt_bit) begin
                            sqrt_operand <= sqrt_operand - sqrt_result - sqrt_bit;
                            sqrt_result <= (sqrt_result >> 1) + sqrt_bit;
                        end else sqrt_result <= sqrt_result >> 1;
                        sqrt_bit <= sqrt_bit >> 2;
                    end
                end
                NORM_START: begin
                    delta = $signed({xmem[index][31], xmem[index]})
                            - $signed(mean[32:0]);
                    norm_div_numerator <= {{47{delta[32]}}, delta, 16'b0};
                    norm_div_denominator <= deviation;
                    norm_div_start <= 1;
                    state <= NORM_WAIT;
                end
                NORM_WAIT: if (norm_div_done) begin
                    normalized <= norm_div_quotient;
                    state <= NORM_AFFINE;
                end
                NORM_AFFINE: begin
                    affine = normalized * $signed(gmem[index]);
                    out_y <= (affine >>> 16) + $signed(bmem[index]);
                    if (index == 0 && debug_capture_active && !debug_capture_done) begin
                        debug_y0 <= (affine >>> 16) + $signed(bmem[index]);
                        debug_normalized0 <= normalized;
                        debug_affine0 <= affine >>> 16;
                        debug_capture_active <= 0;
                        debug_capture_done <= 1;
                    end
                    out_index <= index;
                    out_valid <= 1;
                    state <= EMIT;
                end
                EMIT: if (out_ready) begin
                    out_valid <= 0;
                    if (index == D-1) begin
                        load_count <= 0; sum <= 0; index <= 0; state <= LOAD;
                    end else begin
                        index <= index + 1'b1;
                        state <= NORM_START;
                    end
                end
                default: state <= LOAD;
            endcase
        end
    end
endmodule
