`timescale 1ns / 1ps

// Flat-package signed INT8 GEMV. Matrix metadata comes from the manifest
// controller. IDs 0 (embedding) and 1 (tied head) use the same matrix_base.
module gptneo_resident_gemv #(
    parameter integer MMAX=50257,
    parameter integer KMAX=256,
    parameter integer WDEPTH_BYTES=3852292,
    parameter WEIGHT_FILE=""
) (
    input wire clk,input wire rst,
    input wire weight_we,
    input wire [$clog2((WDEPTH_BYTES+3)/4)-1:0] weight_addr,
    input wire [31:0] weight_data,
    input wire activation_valid,output wire activation_ready,
    input wire signed [7:0] activation_data,
    input wire start,input wire [7:0] matrix_id,
    input wire [$clog2(WDEPTH_BYTES)-1:0] matrix_base,
    input wire [$clog2(KMAX+1)-1:0] row_stride,
    input wire [$clog2(MMAX+1)-1:0] m_count,
    input wire [$clog2(KMAX+1)-1:0] k_count,
    output reg out_valid,input wire out_ready,
    output reg [$clog2(MMAX)-1:0] out_index,
    output reg signed [63:0] out_accumulator,output wire busy
);
    localparam integer WWORDS=(WDEPTH_BYTES+3)/4;
    (* ram_style="block" *) reg [31:0] weights[0:WWORDS-1];
    (* ram_style="distributed" *) reg signed [7:0] activations[0:KMAX-1];
    initial if(WEIGHT_FILE!="") $readmemh(WEIGHT_FILE,weights);
    localparam IDLE=3'd0,READ=3'd1,READ_WAIT=3'd2,MAC=3'd3,EMIT=3'd4;
    reg [2:0] state;
    reg [$clog2(KMAX)-1:0] activation_ptr,column;
    reg [$clog2(MMAX)-1:0] row;
    reg signed [63:0] accumulator;
    reg signed [7:0] activation_q,weight_q;
    reg [$clog2(WDEPTH_BYTES)-1:0] byte_address;
    reg [$clog2(WWORDS)-1:0] word_address;
    reg [1:0] byte_lane;
    reg [31:0] weight_word;
    assign activation_ready=(state==IDLE)&&(activation_ptr<k_count);
    assign busy=(state!=IDLE);
    always @(posedge clk) begin
        if(weight_we) weights[weight_addr]<=weight_data;
        if(rst) begin state<=IDLE;activation_ptr<=0;out_valid<=0;row<=0;column<=0;accumulator<=0; end
        else begin
            if(activation_valid&&activation_ready) begin
                activations[activation_ptr]<=activation_data;activation_ptr<=activation_ptr+1'b1;
            end
            case(state)
                IDLE:if(start) begin row<=0;column<=0;accumulator<=0;out_valid<=0;state<=READ;end
                READ:begin
                    byte_address=matrix_base+row*row_stride+column;
                    word_address<=byte_address>>2;byte_lane<=byte_address[1:0];state<=READ_WAIT;
                end
                READ_WAIT:begin
                    weight_word<=weights[word_address];activation_q<=activations[column];state<=MAC;
                end
                MAC:begin
                    case(byte_lane)
                        0:weight_q=weight_word[7:0];1:weight_q=weight_word[15:8];
                        2:weight_q=weight_word[23:16];default:weight_q=weight_word[31:24];
                    endcase
                    if(column==k_count-1) begin
                        out_accumulator<=accumulator+weight_q*activation_q;out_index<=row;
                        out_valid<=1;state<=EMIT;
                    end else begin accumulator<=accumulator+weight_q*activation_q;column<=column+1'b1;state<=READ;end
                end
                EMIT:if(out_ready) begin
                    out_valid<=0;
                    if(row==m_count-1) begin activation_ptr<=0;state<=IDLE;end
                    else begin row<=row+1'b1;column<=0;accumulator<=0;state<=READ;end
                end
                default:state<=IDLE;
            endcase
        end
    end
endmodule
