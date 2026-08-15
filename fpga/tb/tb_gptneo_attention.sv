`timescale 1ns / 1ps
module tb_gptneo_attention;
    localparam D=64,NHEAD=16,HEAD_DIM=4,TMAX=32;
    reg clk=0; always #5 clk=~clk;
    reg rst=1,cache_reset=0,start=0,in_valid=0,out_ready=0;
    reg [$clog2(TMAX)-1:0] position=0;
    reg signed [31:0] in_q=0,in_k=0,in_v=0;
    wire in_ready,out_valid,busy;
    wire [$clog2(D)-1:0] out_index;
    wire signed [31:0] out_context;
    gptneo_attention dut(.*);
    integer i, lane;
    task begin_token; input integer pos; input integer value; begin
        position=pos; start=1; @(posedge clk); @(negedge clk); start=0;
        for(lane=0;lane<D;lane=lane+1) begin
            while(!in_ready) @(negedge clk);
            in_q=0; in_k=0; in_v=value; in_valid=1;
            @(posedge clk); @(negedge clk); in_valid=0;
        end
    end endtask
    task drain; input integer expected; integer index; integer held; begin
        for(index=0;index<D;index=index+1) begin
            while(!out_valid) @(negedge clk); held=out_context;
            @(negedge clk);
            if(!out_valid || out_context!=held) $fatal(1,"attention backpressure");
            if(out_index!=index || out_context!=expected)
                $fatal(1,"attention idx=%0d got=%0d expected=%0d",out_index,out_context,expected);
            out_ready=1; @(posedge clk); @(negedge clk); out_ready=0;
        end
    end endtask
    initial begin
        repeat(3) @(posedge clk); @(negedge clk); rst=0;
        begin_token(0,65536); drain(65536);
        begin_token(1,3*65536); drain(2*65536);
        for(i=2;i<=15;i=i+1) begin
            begin_token(i,65536); drain(((i+3)*65536+(i+1)/2)/(i+1));
        end
        for(i=16;i<=31;i=i+1) begin
            begin_token(i,65536); drain(((i+3)*65536+(i+1)/2)/(i+1));
        end
        $display("GPTNEO_ATTN_PASS"); $finish;
    end
endmodule
