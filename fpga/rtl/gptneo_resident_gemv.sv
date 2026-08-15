`timescale 1ns / 1ps

// Serial package-resident W8A8 GEMV with Q8.24 scales and Q16.16 outputs.
module gptneo_resident_gemv #(
    parameter integer MMAX=50257, parameter integer KMAX=256,
    parameter integer WDEPTH_BYTES=3825235, parameter WEIGHT_FILE=""
) (
    input wire clk,input wire rst,input wire weight_we,
    input wire [$clog2((WDEPTH_BYTES+3)/4)-1:0] weight_addr,input wire [31:0] weight_data,
    input wire activation_valid,output wire activation_ready,input wire signed [7:0] activation_data,
    input wire start,input wire [7:0] matrix_id,
    input wire [$clog2(WDEPTH_BYTES)-1:0] matrix_base,input wire [$clog2(KMAX+1)-1:0] row_stride,
    input wire [$clog2(WDEPTH_BYTES)-1:0] input_scale_base,
    input wire [$clog2(WDEPTH_BYTES)-1:0] weight_scale_base,
    input wire [$clog2(WDEPTH_BYTES)-1:0] output_scale_base,
    input wire [$clog2(WDEPTH_BYTES)-1:0] bias_base,
    input wire has_output_scale,input wire has_bias,
    input wire [$clog2(MMAX+1)-1:0] m_count,input wire [$clog2(KMAX+1)-1:0] k_count,
    output reg out_valid,input wire out_ready,output reg [$clog2(MMAX)-1:0] out_index,
    output reg signed [63:0] out_accumulator,output reg signed [31:0] out_value_q16,
    output reg signed [7:0] out_code,output wire busy
);
    localparam integer WWORDS=(WDEPTH_BYTES+3)/4;
    (* ram_style="block" *) reg [31:0] image[0:WWORDS-1];
    (* ram_style="distributed" *) reg signed [7:0] activations[0:KMAX-1];
    initial if(WEIGHT_FILE!="") $readmemh(WEIGHT_FILE,image);
    localparam IDLE=6'd0,W_ADDR=6'd1,W_READ=6'd2,AS_ADDR=6'd3,
      AS_READ0=6'd4,AS_READ1=6'd5,AS_READ2=6'd6,AS_CAPTURE=6'd7,MAC=6'd8,
      WS_ADDR=6'd9,WS_READ0=6'd10,WS_READ1=6'd11,WS_READ2=6'd12,WS_CAPTURE=6'd13,
      B_ADDR=6'd14,B_READ=6'd15,B_CAPTURE=6'd16,
      OS_ADDR=6'd17,OS_READ0=6'd18,OS_READ1=6'd19,OS_READ2=6'd20,
      OS_CAPTURE=6'd21,SCALE=6'd22,EMIT=6'd23;
    reg [5:0] state;reg [$clog2(KMAX)-1:0] activation_ptr,column;
    reg [$clog2(MMAX)-1:0] row;reg signed [63:0] accumulator;
    reg signed [7:0] activation_q,weight_q;reg [23:0] activation_scale_q24;
    reg [23:0] weight_scale_q24,output_scale_q24;reg signed [31:0] bias_q16;
    reg [$clog2(WDEPTH_BYTES)-1:0] byte_address;
    reg [$clog2(WWORDS)-1:0] word_address;reg [1:0] byte_lane,scale_lane;
    reg [31:0] read_word,scale_low_word;reg [63:0] scale_window;
    reg signed [127:0] scale_product;reg signed [63:0] real_q16,numerator,qcode_wide;
    reg signed [63:0] magnitude,dequant_product,clipped_code;
    assign activation_ready=(state==IDLE)&&(activation_ptr<k_count);assign busy=(state!=IDLE);
    function automatic signed [63:0] rounded_shift32;
        input signed [127:0] value;reg signed [127:0] mag;begin
            mag=value<0?-value:value;mag=(mag+(128'sd1<<<31))>>>32;
            rounded_shift32=value<0?-mag:mag;end
    endfunction
    always @(posedge clk) begin
        if(weight_we)image[weight_addr]<=weight_data;
        if(rst)begin state<=IDLE;activation_ptr<=0;out_valid<=0;row<=0;column<=0;accumulator<=0;end
        else begin
            if(activation_valid&&activation_ready)begin activations[activation_ptr]<=activation_data;activation_ptr<=activation_ptr+1'b1;end
            case(state)
                IDLE:if(start)begin row<=0;column<=0;accumulator<=0;out_valid<=0;state<=W_ADDR;end
                W_ADDR:begin byte_address=matrix_base+row*row_stride+column;word_address<=byte_address>>2;byte_lane<=byte_address[1:0];state<=W_READ;end
                W_READ:begin read_word<=image[word_address];state<=AS_ADDR;end
                AS_ADDR:begin
                    case(byte_lane)0:weight_q=read_word[7:0];1:weight_q=read_word[15:8];2:weight_q=read_word[23:16];default:weight_q=read_word[31:24];endcase
                    activation_q<=activations[column];byte_address=input_scale_base+column*3;
                    word_address<=byte_address>>2;scale_lane<=byte_address[1:0];state<=AS_READ0;end
                AS_READ0:begin read_word<=image[word_address];state<=AS_READ1;end
                AS_READ1:begin scale_low_word<=read_word;word_address<=word_address+1'b1;state<=AS_READ2;end
                AS_READ2:begin read_word<=image[word_address];state<=AS_CAPTURE;end
                AS_CAPTURE:begin
                    scale_window={read_word,scale_low_word}>>(scale_lane*8);
                    activation_scale_q24<=scale_window[23:0];state<=MAC;end
                MAC:begin
                    accumulator<=accumulator+activation_q*weight_q*$signed({1'b0,activation_scale_q24});
                    if(column==k_count-1)begin column<=0;state<=WS_ADDR;end else begin column<=column+1'b1;state<=W_ADDR;end
                end
                WS_ADDR:begin byte_address=weight_scale_base+row*3;word_address<=byte_address>>2;
                    scale_lane<=byte_address[1:0];state<=WS_READ0;end
                WS_READ0:begin read_word<=image[word_address];state<=WS_READ1;end
                WS_READ1:begin scale_low_word<=read_word;word_address<=word_address+1'b1;state<=WS_READ2;end
                WS_READ2:begin read_word<=image[word_address];state<=WS_CAPTURE;end
                WS_CAPTURE:begin
                    scale_window={read_word,scale_low_word}>>(scale_lane*8);weight_scale_q24<=scale_window[23:0];
                    if(has_bias)state<=B_ADDR;
                    else begin bias_q16<=0;if(has_output_scale)state<=OS_ADDR;else state<=SCALE;end
                end
                B_ADDR:begin word_address<=(bias_base+(row<<2))>>2;state<=B_READ;end
                B_READ:begin read_word<=image[word_address];state<=B_CAPTURE;end
                B_CAPTURE:begin bias_q16<=$signed(read_word);if(has_output_scale)state<=OS_ADDR;else state<=SCALE;end
                OS_ADDR:begin byte_address=output_scale_base+row*3;word_address<=byte_address>>2;
                    scale_lane<=byte_address[1:0];state<=OS_READ0;end
                OS_READ0:begin read_word<=image[word_address];state<=OS_READ1;end
                OS_READ1:begin scale_low_word<=read_word;word_address<=word_address+1'b1;state<=OS_READ2;end
                OS_READ2:begin read_word<=image[word_address];state<=OS_CAPTURE;end
                OS_CAPTURE:begin scale_window={read_word,scale_low_word}>>(scale_lane*8);
                    output_scale_q24<=scale_window[23:0];state<=SCALE;end
                SCALE:begin
                    scale_product=$signed(accumulator)*$signed({1'b0,weight_scale_q24});
                    real_q16=rounded_shift32(scale_product)+$signed(bias_q16);out_accumulator<=accumulator;
                    if(has_output_scale)begin
                        numerator=real_q16<<<8;magnitude=numerator<0?-numerator:numerator;
                        qcode_wide=(magnitude+(output_scale_q24>>1))/$signed({1'b0,output_scale_q24});
                        if(numerator<0)qcode_wide=-qcode_wide;
                        if(qcode_wide>127)clipped_code=127;else if(qcode_wide< -128)clipped_code=-128;else clipped_code=qcode_wide;
                        out_code<=clipped_code[7:0];dequant_product=clipped_code*$signed({1'b0,output_scale_q24});
                        magnitude=dequant_product<0?-dequant_product:dequant_product;
                        magnitude=(magnitude+128)>>>8;
                        out_value_q16<=dequant_product<0?-magnitude:magnitude;
                    end else begin out_code<=0;out_value_q16<=real_q16;end
                    out_index<=row;out_valid<=1;state<=EMIT;
                end
                EMIT:if(out_ready)begin out_valid<=0;if(row==m_count-1)begin activation_ptr<=0;state<=IDLE;end
                    else begin row<=row+1'b1;accumulator<=0;state<=W_ADDR;end end
                default:state<=IDLE;
            endcase
        end
    end
endmodule
