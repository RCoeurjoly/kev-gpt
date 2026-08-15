`timescale 1ns / 1ps

module tb_gptneo_gelu;
    reg clk = 0;
    always #5 clk = ~clk;
    reg rst = 1;
    reg in_valid = 0;
    wire in_ready;
    reg signed [15:0] in_data = 0;
    wire out_valid;
    reg out_ready = 0;
    wire signed [15:0] out_data;

    gptneo_gelu dut (.*);

    task check;
        input signed [15:0] value;
        input signed [15:0] expected;
        integer held;
        begin
            while (!in_ready) @(posedge clk);
            in_data <= value;
            in_valid <= 1;
            @(posedge clk);
            in_valid <= 0;
            while (!out_valid) @(posedge clk);
            held = out_data;
            repeat (3) begin
                @(posedge clk);
                if (!out_valid || out_data !== held) $fatal(1, "GELU backpressure failure");
            end
            if (out_data !== expected)
                $fatal(1, "GELU mismatch x=%0d got=%0d expected=%0d", value, out_data, expected);
            out_ready <= 1;
            @(posedge clk);
            out_ready <= 0;
        end
    endtask

    initial begin
        repeat (3) @(posedge clk);
        rst <= 0;
        check(-16'sd32768, 16'sd0);
        check(16'sd0, 16'sd0);
        check(16'sd4096, 16'sd3446);
        check(16'sd16384, 16'sd16384);
        $display("GPTNEO_GELU_PASS");
        $finish;
    end
endmodule
