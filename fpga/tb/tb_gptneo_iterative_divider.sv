`timescale 1ns / 1ps
module tb_gptneo_iterative_divider;
    reg clk=0; always #5 clk=~clk;
    reg rst=1,start=0;
    reg signed [95:0] numerator=0;
    reg [63:0] denominator=1;
    wire busy,done;
    wire signed [31:0] quotient;
    gptneo_iterative_divider dut(.*);

    task check;
        input signed [95:0] n;
        input [63:0] d;
        input signed [31:0] expected;
        integer cycles;
        begin
            while(busy) @(negedge clk);
            numerator=n; denominator=d; start=1;
            @(posedge clk); @(negedge clk); start=0;
            cycles=0;
            while(!done) begin
                @(negedge clk); cycles=cycles+1;
                if(cycles>100) $fatal(1,"divider timeout");
            end
            if(quotient!==expected)
                $fatal(1,"divider n=%0d d=%0d got=%0d expected=%0d",n,d,quotient,expected);
        end
    endtask

    initial begin
        repeat(3) @(posedge clk); @(negedge clk); rst=0;
        check(0,1,0);
        check(96'sd131072,64'd2,32'sd65536);
        check(-96'sd131073,64'd2,-32'sd65536);
        check(96'sd2147483647,64'd1048576,32'sd2047);
        check(-96'sd2147483647,64'd1048576,-32'sd2047);
        check(96'd9223372030412324865,64'h00000000ffffffff,32'sh7fffffff);
        $display("GPTNEO_DIVIDER_PASS"); $finish;
    end
endmodule
