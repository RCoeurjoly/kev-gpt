`timescale 1ns / 1ps
module tb_gptneo_sequencer;
`include "gptneo_package.svh"
  reg clk=0;always #1 clk=~clk;
  reg rst=1,clear=0,prompt_valid=0,start=0,token_ready=1;
  reg [15:0] prompt_token=0;reg [5:0] requested_tokens=0;reg [31:0] package_tag=0;
  wire prompt_ready,token_valid,busy,error;wire [15:0] token_id;wire [7:0] error_code;
  wire [62:0] debug_status;
  wire [95:0] debug_embedding;
  wire [95:0] debug_layernorm;
  wire memory_req_valid,memory_rsp_ready;
  reg memory_req_ready=0,memory_rsp_valid=0;
  wire [$clog2((3825235+3)/4)-1:0] memory_req_word_addr;
  reg [31:0] memory_rsp_data=0;
  reg [15:0] prompts[0:63],expected[0:63];
  integer observed;
`ifdef GPTNEO_DEBUG_EMB
  integer debug_i;reg debug_seen=0;
  always @(posedge clk) if(!debug_seen && dut.state==dut.LN_G_REQ && dut.layer==0 &&
    dut.position==0 && dut.index==0 && !dut.ln_is_second && !dut.ln_is_final)begin
    debug_seen=1;$display("DEBUG_EMBEDDING %024x",debug_embedding);
    for(debug_i=0;debug_i<64;debug_i=debug_i+1)
      $display("DEBUG_X %0d %0d",debug_i,dut.xmem[debug_i]);
    $finish;
  end
`endif
`ifdef GPTNEO_DEBUG_SECOND_SCORE
  integer debug_s_i;reg debug_s_seen=0;
  always @(posedge clk) if(!debug_s_seen && dut.generated_count==1 && dut.layer==0 &&
    dut.position==dut.token_count-1 && dut.attn_out_valid && dut.attn_out_index==3)begin
    debug_s_seen=1;$display("DEBUG_EXPSUM %0d",dut.attention.exp_sum);
    $display("DEBUG_QHEAD %0d %0d %0d %0d",dut.attention.qmem[0],
      dut.attention.qmem[1],dut.attention.qmem[2],dut.attention.qmem[3]);
    for(debug_s_i=0;debug_s_i<5;debug_s_i=debug_s_i+1)
      $display("DEBUG_SCORE %0d %0d EXP=%0d V=%0d K=%0d,%0d,%0d,%0d",debug_s_i,
        dut.attention.scoremem[debug_s_i],dut.attention.expmem[debug_s_i],
        dut.attention.vmem[debug_s_i*64],dut.attention.kmem[debug_s_i*64],
        dut.attention.kmem[debug_s_i*64+1],dut.attention.kmem[debug_s_i*64+2],
        dut.attention.kmem[debug_s_i*64+3]);
    $finish;
  end
`endif
`ifdef GPTNEO_DEBUG_SECOND_CTX
  integer debug_ctx_i;reg debug_ctx_seen=0;
  always @(posedge clk) if(!debug_ctx_seen && dut.generated_count==1 &&
    dut.state==dut.GEMV_LOAD && dut.gemv_op==dut.OP_OUT && dut.layer==0 &&
    dut.position==dut.token_count-1)begin
    debug_ctx_seen=1;for(debug_ctx_i=0;debug_ctx_i<64;debug_ctx_i=debug_ctx_i+1)
      $display("DEBUG_CTX %0d %0d",debug_ctx_i,dut.ctxbuf[debug_ctx_i]);
    $finish;
  end
`endif
`ifdef GPTNEO_DEBUG_SECOND_Q
  integer debug_q_i;reg debug_q_seen=0;
  always @(posedge clk) if(!debug_q_seen && dut.generated_count==1 &&
    dut.state==dut.GEMV_LOAD && dut.gemv_op==dut.OP_K && dut.layer==0 &&
    dut.position==dut.token_count-1)begin
    debug_q_seen=1;for(debug_q_i=0;debug_q_i<64;debug_q_i=debug_q_i+1)
      $display("DEBUG_Q %0d %0d",debug_q_i,dut.qbuf[debug_q_i]);
    $finish;
  end
`endif
`ifdef GPTNEO_DEBUG_SECOND_LAYER
  integer debug_layer_i;reg debug_layer_seen=0;
  always @(posedge clk) if(!debug_layer_seen && dut.generated_count==1 &&
    dut.state==dut.NEXT_POSITION && dut.layer==0 && dut.position==dut.token_count-1)begin
    debug_layer_seen=1;
    $display("DEBUG_TOKENS count=%0d %0d %0d %0d %0d",dut.token_count,dut.tokens[0],
      dut.tokens[1],dut.tokens[2],dut.tokens[3],dut.tokens[4]);
    for(debug_layer_i=0;debug_layer_i<64;debug_layer_i=debug_layer_i+1)
      $display("DEBUG_LAYER0 %0d %0d",debug_layer_i,
        dut.xmem[dut.position*64+debug_layer_i]);
    $finish;
  end
`endif
`ifdef GPTNEO_DEBUG_LN
  integer debug_ln_i;reg debug_ln_seen=0;
  always @(posedge clk) if(!debug_ln_seen && dut.state==dut.GEMV_LOAD &&
    dut.gemv_op==dut.OP_Q && dut.layer==0 && dut.position==0)begin
    debug_ln_seen=1;$display("DEBUG_LN_STATS mean=%0d deviation=%0d variance=%0d",
      dut.layernorm.mean,dut.layernorm.deviation,dut.layernorm.variance);
    for(debug_ln_i=0;debug_ln_i<64;debug_ln_i=debug_ln_i+1)
      $display("DEBUG_LN %0d %0d X=%0d G=%0d B=%0d",debug_ln_i,dut.norm[debug_ln_i],
        dut.layernorm.xmem[debug_ln_i],dut.layernorm.gmem[debug_ln_i],dut.layernorm.bmem[debug_ln_i]);
    $finish;
  end
`endif
`ifdef GPTNEO_CHECK_LN_FEED
  reg ln_feed_checked=0;
  always @(posedge clk) if(!ln_feed_checked && dut.state==dut.LN_START &&
    dut.layer==0 && dut.position==0 && !dut.ln_is_second && !dut.ln_is_final)begin
    ln_feed_checked=1;
    if(dut.layernorm.xmem[0]!==dut.xmem[0])
      $fatal(1,"LN_FEED_MISMATCH captured=%0d expected=%0d",
        dut.layernorm.xmem[0],dut.xmem[0]);
    $display("LN_FEED_MATCH value=%0d",dut.xmem[0]);$finish;
  end
`endif
  gptneo_sequencer dut(.*);

  task reset_case;begin clear<=1;@(posedge clk);clear<=0;repeat(2)@(posedge clk);end endtask
  task put_prompt;input integer offset;input integer length;integer i;begin
    for(i=0;i<length;i=i+1)begin while(!prompt_ready)@(posedge clk);
      prompt_token<=prompts[offset+i];prompt_valid<=1;@(posedge clk);prompt_valid<=0;end
  end endtask
  task run_case;input integer case_id;input integer poff;input integer plen;
    input integer eoff;input integer elen;integer i;begin
    reset_case;put_prompt(poff,plen);requested_tokens<=elen;package_tag<=GPTNEO_PACKAGE_TAG;
    start<=1;@(posedge clk);start<=0;observed=0;
    for(i=0;i<elen;i=i+1)begin
      while(!token_valid)begin @(posedge clk);if(error)$fatal(1,"sequencer error %0d",error_code);end
      if(token_id!==expected[eoff+i])$fatal(1,"case=%0d token=%0d got=%0d expected=%0d",
        case_id,i,token_id,expected[eoff+i]);
      observed=observed+1;@(posedge clk);
      $display("GPTNEO_SEQ_PROGRESS case=%0d token=%0d/%0d",case_id,observed,elen);
    end
    while(busy)@(posedge clk);
    if(case_id==0 && debug_embedding!==96'h690046dd000027dd000044ee)
      $fatal(1,"case=0 embedding probe got=%024x",debug_embedding);
    if(case_id==0 && debug_layernorm!==96'h00021c770001eb4300021048)
      $fatal(1,"case=0 layernorm probe got=%024x",debug_layernorm);
    $display("GPTNEO_SEQ_PASS case=%0d tokens=%0d/%0d",case_id,observed,elen);
  end endtask
  initial begin
    $readmemh("prompt_tokens.mem",prompts);$readmemh("expected_tokens.mem",expected);
    repeat(4)@(posedge clk);rst<=0;repeat(2)@(posedge clk);
    package_tag<=~GPTNEO_PACKAGE_TAG;requested_tokens<=1;start<=1;@(posedge clk);start<=0;
    while(!error)@(posedge clk);if(error_code!==8'd1)$fatal(1,"wrong package error");
    $display("GPTNEO_SEQ_NEGATIVE_PASS error=PACKAGE_HASH");
    run_case(0,GPTNEO_CASE_0_PROMPT_OFFSET,GPTNEO_CASE_0_PROMPT_LENGTH,
      GPTNEO_CASE_0_EXPECTED_OFFSET,GPTNEO_CASE_0_EXPECTED_LENGTH);
    run_case(1,GPTNEO_CASE_1_PROMPT_OFFSET,GPTNEO_CASE_1_PROMPT_LENGTH,
      GPTNEO_CASE_1_EXPECTED_OFFSET,GPTNEO_CASE_1_EXPECTED_LENGTH);
    run_case(2,GPTNEO_CASE_2_PROMPT_OFFSET,GPTNEO_CASE_2_PROMPT_LENGTH,
      GPTNEO_CASE_2_EXPECTED_OFFSET,GPTNEO_CASE_2_EXPECTED_LENGTH);
    $finish;
  end
endmodule
