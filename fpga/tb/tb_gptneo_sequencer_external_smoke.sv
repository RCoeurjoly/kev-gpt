`timescale 1ns / 1ps

module tb_gptneo_sequencer_external_smoke;
`include "gptneo_package.svh"
    localparam WDEPTH_BYTES=3825235;
    localparam WORD_ADDR_BITS=$clog2((WDEPTH_BYTES+3)/4);
    localparam [WORD_ADDR_BITS-1:0] EMBEDDING_WORD=
        GPTNEO_TENSOR_TOKEN_EMBEDDING_WEIGHT_OFFSET>>2;
    localparam [WORD_ADDR_BITS-1:0] TOKEN_SCALE_WORD=
        (GPTNEO_SCALE_BASE+13920)>>2;

    reg clk=0;always #1 clk=~clk;
    reg rst=1,clear=0,prompt_valid=0,start=0,token_ready=1;
    reg [15:0] prompt_token=0;
    reg [5:0] requested_tokens=1;
    reg [31:0] package_tag=GPTNEO_PACKAGE_TAG;
    wire prompt_ready,token_valid,busy,error;
    wire [15:0] token_id;
    wire [7:0] error_code;
    wire [62:0] debug_status;
    wire [95:0] debug_embedding,debug_layernorm;

    wire memory_req_valid;
    reg memory_req_ready=1;
    wire [WORD_ADDR_BITS-1:0] memory_req_word_addr;
    reg memory_rsp_valid=0;
    wire memory_rsp_ready;
    reg [31:0] memory_rsp_data=0;
    integer request_count=0,response_delay=-1,timeout=0;

    gptneo_sequencer #(
        .WDEPTH_BYTES(WDEPTH_BYTES),.EXTERNAL_MEMORY(1)
    ) dut (
        .clk(clk),.rst(rst),.clear(clear),
        .prompt_valid(prompt_valid),.prompt_ready(prompt_ready),.prompt_token(prompt_token),
        .start(start),.requested_tokens(requested_tokens),.package_tag(package_tag),
        .token_valid(token_valid),.token_ready(token_ready),.token_id(token_id),
        .busy(busy),.error(error),.error_code(error_code),
        .debug_status(debug_status),.debug_embedding(debug_embedding),
        .debug_layernorm(debug_layernorm),
        .memory_req_valid(memory_req_valid),.memory_req_ready(memory_req_ready),
        .memory_req_word_addr(memory_req_word_addr),
        .memory_rsp_valid(memory_rsp_valid),.memory_rsp_ready(memory_rsp_ready),
        .memory_rsp_data(memory_rsp_data)
    );

    always @(posedge clk) begin
        timeout<=timeout+1;
        if(timeout>500)$fatal(1,"sequencer external-memory request timeout");
        if(memory_req_valid&&memory_req_ready)begin
            case(request_count)
                0:if(memory_req_word_addr!==EMBEDDING_WORD)
                    $fatal(1,"first embedding word address mismatch");
                1:if(memory_req_word_addr!==EMBEDDING_WORD+1'b1)
                    $fatal(1,"second embedding word address mismatch");
                2:begin
                    if(memory_req_word_addr!==TOKEN_SCALE_WORD)
                        $fatal(1,"token scale word address mismatch");
                    $display("GPTNEO_SEQUENCER_EXTERNAL_PASS requests=3");
                    $finish;
                end
            endcase
            request_count<=request_count+1;
            response_delay<=2;
        end
        if(response_delay>0)response_delay<=response_delay-1;
        else if(response_delay==0&&!memory_rsp_valid)begin
            memory_rsp_data<=0;memory_rsp_valid<=1;
        end
        if(memory_rsp_valid&&memory_rsp_ready)begin
            memory_rsp_valid<=0;response_delay<=-1;
        end
    end

    initial begin
        repeat(4)@(posedge clk);rst<=0;repeat(2)@(posedge clk);
        while(!prompt_ready)@(posedge clk);
        prompt_valid<=1;@(posedge clk);prompt_valid<=0;
        start<=1;@(posedge clk);start<=0;
    end
endmodule
