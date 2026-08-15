`timescale 1ns / 1ps
module tinystories_selftest_top(
  input wire SYS_CLK,input wire SYS_RSTN,output wire [2:0] led_3bits_tri_o
);
`include "gptneo_package.svh"
  localparam LOAD=3'd0,LAUNCH=3'd1,WAIT=3'd2,DONE=3'd3;
  reg [2:0] state=LOAD;reg [2:0] prompt_index=0;reg prompt_valid=0,start=0;
  reg [15:0] prompt_token;wire prompt_ready,token_valid,busy,error;
  wire [15:0] token_id;wire [7:0] error_code;reg pass=0,fail=0;
  reg [25:0] heartbeat=0;
  wire reset=!SYS_RSTN;
  function automatic [15:0] prompt_word;input [2:0] which;begin
    case(which) 0:prompt_word=16'd7454;1:prompt_word=16'd2402;
      2:prompt_word=16'd257;default:prompt_word=16'd640;endcase
  end endfunction
  gptneo_sequencer decoder(.clk(SYS_CLK),.rst(reset),.clear(1'b0),
    .prompt_valid(prompt_valid),.prompt_ready(prompt_ready),.prompt_token(prompt_token),
    .start(start),.requested_tokens(6'd1),.package_tag(GPTNEO_PACKAGE_TAG),
    .token_valid(token_valid),.token_ready(1'b1),.token_id(token_id),.busy(busy),
    .error(error),.error_code(error_code));
  always @(posedge SYS_CLK)begin
    heartbeat<=heartbeat+1'b1;prompt_valid<=0;start<=0;
    if(reset)begin state<=LOAD;prompt_index<=0;pass<=0;fail<=0;heartbeat<=0;end
    else case(state)
      LOAD:if(prompt_ready)begin prompt_token<=prompt_word(prompt_index);prompt_valid<=1;
        if(prompt_index==3)state<=LAUNCH;else prompt_index<=prompt_index+1'b1;end
      LAUNCH:begin start<=1;state<=WAIT;end
      WAIT:if(error)begin fail<=1;state<=DONE;end else if(token_valid)begin
        if(token_id==16'd11)pass<=1;else fail<=1;state<=DONE;end
      default:state<=DONE;
    endcase
  end
  assign led_3bits_tri_o={fail,pass,heartbeat[25]};
endmodule
