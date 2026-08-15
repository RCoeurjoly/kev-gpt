`timescale 1ns / 1ps

// One quotient bit per cycle. The attention numerator is signed Q16.36-ish,
// while the softmax denominator is positive. Division truncates toward zero,
// matching SystemVerilog signed division without creating a combinational
// divider in the FPGA netlist.
module gptneo_iterative_divider (
    input wire clk,
    input wire rst,
    input wire start,
    input wire signed [95:0] numerator,
    input wire [63:0] denominator,
    output reg busy,
    output reg done,
    output reg signed [31:0] quotient
);
    reg negative;
    reg [95:0] dividend;
    reg [63:0] divisor;
    reg [64:0] remainder;
    reg [95:0] quotient_work;
    reg [6:0] bit_index;
    reg [64:0] shifted_remainder;
    reg [95:0] completed_quotient;

    always @(posedge clk) begin
        done <= 1'b0;
        if (rst) begin
            busy <= 1'b0;
            quotient <= 0;
            remainder <= 0;
            quotient_work <= 0;
            bit_index <= 0;
        end else if (start && !busy) begin
            if (denominator == 0) begin
                quotient <= 0;
                done <= 1'b1;
            end else begin
                negative <= numerator[95];
                dividend <= numerator[95] ? -numerator : numerator;
                divisor <= denominator;
                remainder <= 0;
                quotient_work <= 0;
                bit_index <= 7'd95;
                busy <= 1'b1;
            end
        end else if (busy) begin
            shifted_remainder = {remainder[63:0], dividend[bit_index]};
            completed_quotient = quotient_work;
            if (shifted_remainder >= {1'b0, divisor}) begin
                remainder <= shifted_remainder - {1'b0, divisor};
                completed_quotient[bit_index] = 1'b1;
                quotient_work[bit_index] <= 1'b1;
            end else begin
                remainder <= shifted_remainder;
            end

            if (bit_index == 0) begin
                quotient <= negative ? -$signed(completed_quotient[31:0])
                                     :  $signed(completed_quotient[31:0]);
                busy <= 1'b0;
                done <= 1'b1;
            end else begin
                bit_index <= bit_index - 1'b1;
            end
        end
    end
endmodule
