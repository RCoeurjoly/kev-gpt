module tinystories_interactive_top(
  input wire SYS_CLK,input wire SYS_RSTN,output wire [2:0] LED
);
`include "gptneo_package.svh"
  wire reset=!SYS_RSTN;
  reg [25:0] heartbeat=0;reg failure=0;
  wire rx_valid,rx_ready,tx_valid,tx_ready;wire [7:0] rx_data,tx_data;
  wire seq_clear,prompt_valid,prompt_ready,seq_start,seq_token_valid;
  wire seq_token_ready,seq_busy,seq_error;wire [15:0] prompt_token,seq_token_id;
  wire [5:0] requested_tokens;wire [7:0] seq_error_code;
  always @(posedge SYS_CLK)begin
    if(reset)begin heartbeat<=0;failure<=0;end
    else begin heartbeat<=heartbeat+1'b1;if(seq_error)failure<=1;end
  end
  assign LED={failure,seq_busy,heartbeat[25]};

  bscan_packet_endpoint transport(
    .sys_clk(SYS_CLK),.sys_rst(reset),.rx_valid(rx_valid),.rx_ready(rx_ready),
    .rx_data(rx_data),.tx_valid(tx_valid),.tx_ready(tx_ready),.tx_data(tx_data));
  tinystories_packet_controller controller(
    .clk(SYS_CLK),.rst(reset),.rx_valid(rx_valid),.rx_ready(rx_ready),.rx_data(rx_data),
    .tx_valid(tx_valid),.tx_ready(tx_ready),.tx_data(tx_data),
    .seq_clear(seq_clear),.prompt_valid(prompt_valid),.prompt_ready(prompt_ready),
    .prompt_token(prompt_token),.seq_start(seq_start),.requested_tokens(requested_tokens),
    .seq_token_valid(seq_token_valid),.seq_token_ready(seq_token_ready),
    .seq_token_id(seq_token_id),.seq_busy(seq_busy),.seq_error(seq_error),
    .seq_error_code(seq_error_code));
  gptneo_sequencer sequencer(
    .clk(SYS_CLK),.rst(reset),.clear(seq_clear),.prompt_valid(prompt_valid),
    .prompt_ready(prompt_ready),.prompt_token(prompt_token),.start(seq_start),
    .requested_tokens(requested_tokens),.package_tag(GPTNEO_PACKAGE_TAG),
    .token_valid(seq_token_valid),.token_ready(seq_token_ready),.token_id(seq_token_id),
    .busy(seq_busy),.error(seq_error),.error_code(seq_error_code));
endmodule
