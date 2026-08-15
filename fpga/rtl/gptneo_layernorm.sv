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
    output wire busy
);
    localparam [63:0] EPSILON_Q32 = 64'd42950;
    localparam LOAD=4'd0, MEAN=4'd1, VAR=4'd2, SQRT_ALIGN=4'd3,
               SQRT_STEP=4'd4, NORM=4'd5, EMIT=4'd6;
    reg [3:0] state;
    reg signed [31:0] xmem[0:D-1], gmem[0:D-1], bmem[0:D-1];
    reg [$clog2(D+1)-1:0] load_count;
    reg [$clog2(D)-1:0] index;
    reg signed [63:0] sum;
    reg signed [63:0] mean;
    reg [79:0] square_sum;
    reg signed [63:0] delta;
    reg [63:0] delta_magnitude;
    reg [127:0] delta_square;
    reg [63:0] variance;
    reg [63:0] deviation;
    reg signed [95:0] normalized;
    reg signed [127:0] affine;
    reg [63:0] sqrt_operand, sqrt_result, sqrt_bit;

    assign in_ready = (state == LOAD) && (load_count < D);
    assign busy = (state != LOAD) || (load_count != 0);

    always @(posedge clk) begin
        if (rst) begin
            state <= LOAD; load_count <= 0; sum <= 0; out_valid <= 0;
            index <= 0; square_sum <= 0;
        end else begin
            if (in_valid && in_ready) begin
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
                    delta = $signed(xmem[index]) - mean;
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
                        state <= NORM;
                    end else begin
                        if (sqrt_operand >= sqrt_result + sqrt_bit) begin
                            sqrt_operand <= sqrt_operand - sqrt_result - sqrt_bit;
                            sqrt_result <= (sqrt_result >> 1) + sqrt_bit;
                        end else sqrt_result <= sqrt_result >> 1;
                        sqrt_bit <= sqrt_bit >> 2;
                    end
                end
                NORM: begin
                    delta = $signed(xmem[index]) - mean;
                    normalized = (delta <<< 16) / $signed(deviation);
                    affine = normalized * $signed(gmem[index]);
                    out_y <= (affine >>> 16) + $signed(bmem[index]);
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
                        state <= NORM;
                    end
                end
                default: state <= LOAD;
            endcase
        end
    end
endmodule
