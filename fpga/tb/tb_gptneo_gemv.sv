`timescale 1ns / 1ps
module tb_gptneo_gemv;
    localparam MMAX=4,KMAX=4,WDEPTH_BYTES=64;
    reg clk=0;always #5 clk=~clk;
    reg rst=1,weight_we=0,activation_valid=0,start=0,out_ready=0;
    reg [$clog2((WDEPTH_BYTES+3)/4)-1:0] weight_addr=0;reg [31:0] weight_data=0;
    reg signed [7:0] activation_data=0;wire activation_ready,out_valid,busy;
    reg [7:0] matrix_id=0;reg [$clog2(WDEPTH_BYTES)-1:0] matrix_base=0;
    reg [$clog2(WDEPTH_BYTES)-1:0] input_scale_base=20,weight_scale_base=32;
    reg [$clog2(WDEPTH_BYTES)-1:0] output_scale_base=40,bias_base=48;
    reg has_output_scale=1,has_bias=0;
    reg [$clog2(KMAX+1)-1:0] row_stride=KMAX,k_count=0;
    reg [$clog2(MMAX+1)-1:0] m_count=0;
    wire [$clog2(MMAX)-1:0] out_index;wire signed [63:0] out_accumulator;
    wire signed [31:0] out_value_q16;wire signed [7:0] out_code;
    gptneo_resident_gemv #(.MMAX(MMAX),.KMAX(KMAX),.WDEPTH_BYTES(WDEPTH_BYTES)) dut(.*);
    task putw;input integer address;input [31:0] value;begin
        weight_addr<=address;weight_data<=value;weight_we<=1;@(posedge clk);weight_we<=0;end endtask
    task putx;input integer value;begin while(!activation_ready)@(posedge clk);
        activation_data<=value;activation_valid<=1;@(posedge clk);activation_valid<=0;end endtask
    task launch;input integer id;input integer base;input integer rows;begin
        matrix_id<=id;matrix_base<=base;m_count<=rows;k_count<=3;start<=1;@(posedge clk);start<=0;end endtask
    task check_output;input integer index;input integer value;integer held;begin
        while(!out_valid)@(negedge clk);held=out_accumulator;repeat(2)begin @(negedge clk);
        if(!out_valid||$signed(out_accumulator)!=$signed(held))$fatal(1,"GEMV backpressure");end
        if(out_index!==index||out_code!==value||out_value_q16!==value*16384)
            $fatal(1,"GEMV idx=%0d code=%0d q16=%0d acc=%0d ws=%0d os=%0d",
                out_index,out_code,out_value_q16,dut.accumulator,dut.weight_scale_q24,dut.output_scale_q24);
        out_ready=1;@(posedge clk);@(negedge clk);out_ready=0;end endtask
    initial begin repeat(3)@(posedge clk);rst<=0;
        putw(0,{8'd0,8'd3,-8'sd2,8'd1});putw(1,{8'd0,-8'sd6,8'd5,-8'sd4});
        putw(4,{8'd0,8'd9,8'd8,8'd7});
        // Three-byte little-endian Q8.24 scales intentionally cross word boundaries.
        putw(5,32'h00800000);putw(6,32'h00008000);putw(7,32'h00000080);
        putw(8,32'h00800000);putw(9,32'h00008000);
        putw(10,32'h00400000);putw(11,32'h00004000);
        k_count<=3;putx(2);putx(-1);putx(3);launch(0,0,2);check_output(0,13);check_output(1,-31);
        putx(2);putx(-1);putx(3);launch(1,0,1);check_output(0,13);
        putx(2);putx(-1);putx(3);launch(2,16,1);check_output(0,33);
        while(busy)@(posedge clk);$display("GPTNEO_GEMV_PASS");$finish;end
endmodule
