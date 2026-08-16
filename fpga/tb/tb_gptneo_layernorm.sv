`timescale 1ns / 1ps

module tb_gptneo_layernorm;
    localparam D=4;
    reg clk=0; always #5 clk=~clk;
    reg rst=1,in_valid=0,start=0,out_ready=0;
    reg signed [31:0] in_x=0,in_gamma=0,in_beta=0;
    wire in_ready,out_valid,busy;
    wire [$clog2(D)-1:0] out_index;
    wire signed [31:0] out_y;
    wire [95:0] debug_status;
    gptneo_layernorm #(.D(D)) dut(.*);

    task put; input integer x; input integer gamma; input integer beta; begin
        while(!in_ready) @(negedge clk);
        in_x=x; in_gamma=gamma; in_beta=beta; in_valid=1;
        @(posedge clk); @(negedge clk); in_valid=0;
    end endtask
    task go; begin start=1; @(posedge clk); @(negedge clk); start=0; end endtask
    task check; input integer idx; input integer expected; integer held; begin
        while(!out_valid) @(negedge clk); held=out_y;
        repeat(2) begin @(negedge clk); if(!out_valid || out_y!=held) $fatal(1,"LayerNorm backpressure"); end
        if(out_index!=idx || out_y!=expected) $fatal(1,"LayerNorm idx=%0d got=%0d expected=%0d",out_index,out_y,expected);
        out_ready=1; @(posedge clk); @(negedge clk); out_ready=0;
    end endtask

    initial begin
        repeat(3) @(posedge clk); @(negedge clk); rst=0;
        put(5*65536,65536,10); put(5*65536,65536,20);
        put(5*65536,65536,30); put(5*65536,65536,40); go;
        check(0,10); check(1,20); check(2,30); check(3,40);
        put(-65536,2*65536,0); put(65536,2*65536,0);
        put(-65536,2*65536,0); put(65536,2*65536,0); go;
        check(0,-131072); check(1,131072); check(2,-131072); check(3,131072);
        $display("GPTNEO_LAYERNORM_PASS"); $finish;
    end
endmodule
