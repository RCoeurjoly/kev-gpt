`timescale 1ns / 1ps

module tb_gptneo_external_memory;
    localparam MMAX=4,KMAX=4,WDEPTH_BYTES=64;
    localparam WORD_ADDR_BITS=$clog2((WDEPTH_BYTES+3)/4);
    reg clk=0;always #5 clk=~clk;
    reg rst=1;

    reg raw_read=0,raw_ready=0;
    reg [$clog2(WDEPTH_BYTES)-1:0] raw_addr=0;
    wire raw_valid;
    wire [31:0] raw_data;

    wire memory_req_valid;
    reg memory_req_ready=0;
    wire [WORD_ADDR_BITS-1:0] memory_req_word_addr;
    reg memory_rsp_valid=0;
    wire memory_rsp_ready;
    reg [31:0] memory_rsp_data=0;

    integer request_count=0;
    reg [WORD_ADDR_BITS-1:0] accepted_addr=0;
    integer response_delay=-1;

    gptneo_resident_gemv #(
        .MMAX(MMAX),.KMAX(KMAX),.WDEPTH_BYTES(WDEPTH_BYTES),
        .EXTERNAL_MEMORY(1)
    ) dut (
        .clk(clk),.rst(rst),.weight_we(1'b0),.weight_addr('0),.weight_data('0),
        .raw_read(raw_read),.raw_addr(raw_addr),.raw_valid(raw_valid),
        .raw_ready(raw_ready),.raw_data(raw_data),
        .activation_valid(1'b0),.activation_ready(),.activation_data_q16('0),
        .start(1'b0),.matrix_id('0),.matrix_base('0),.row_stride('0),
        .input_scale_base('0),.weight_scale_base('0),.output_scale_base('0),
        .bias_base('0),.has_output_scale(1'b0),.has_bias(1'b0),
        .m_count('0),.k_count('0),.out_valid(),.out_ready(1'b0),.out_index(),
        .out_accumulator(),.out_value_q16(),.out_code(),.busy(),
        .memory_req_valid(memory_req_valid),.memory_req_ready(memory_req_ready),
        .memory_req_word_addr(memory_req_word_addr),
        .memory_rsp_valid(memory_rsp_valid),.memory_rsp_ready(memory_rsp_ready),
        .memory_rsp_data(memory_rsp_data)
    );

    // A real ready/valid responder: each request is initially backpressured,
    // then answered after a different non-zero latency.
    always @(posedge clk) begin
        memory_req_ready<=0;
        if(memory_req_valid&&response_delay<0)begin
            memory_req_ready<=1;
            accepted_addr<=memory_req_word_addr;
            response_delay<=request_count==0?3:5;
            request_count<=request_count+1;
        end
        if(response_delay>0)response_delay<=response_delay-1;
        else if(response_delay==0&&!memory_rsp_valid)begin
            memory_rsp_valid<=1;
            memory_rsp_data<=accepted_addr==0?32'h44332211:32'h88776655;
        end
        if(memory_rsp_valid&&memory_rsp_ready)begin memory_rsp_valid<=0;response_delay<=-1;end
    end

    initial begin
        repeat(3)@(posedge clk);rst<=0;
        raw_addr<=1;raw_read<=1;@(posedge clk);raw_read<=0;
        while(!raw_valid)@(negedge clk);
        if(raw_data!==32'h55443322)$fatal(1,"external raw read mismatch: %08x",raw_data);
        if(request_count!==2)$fatal(1,"expected two word requests, got %0d",request_count);
        raw_ready<=1;@(posedge clk);raw_ready<=0;
        $display("GPTNEO_EXTERNAL_MEMORY_PASS requests=%0d",request_count);
        $finish;
    end
endmodule
