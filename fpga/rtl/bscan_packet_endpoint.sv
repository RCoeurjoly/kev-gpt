module bscan_packet_endpoint(
  input wire sys_clk,input wire sys_rst,
  output wire rx_valid,input wire rx_ready,output wire [7:0] rx_data,
  input wire tx_valid,output wire tx_ready,input wire [7:0] tx_data
`ifdef BSCAN_SIM
  ,input wire scan_drck,input wire scan_reset,input wire scan_sel,
  input wire scan_shift,input wire scan_capture,input wire scan_update,
  input wire scan_tdi,output wire scan_tdo
`endif
);
`ifndef BSCAN_SIM
  wire scan_drck,scan_reset,scan_sel,scan_shift,scan_capture,scan_update,scan_tdi;
  wire scan_tdo;
  BSCANE2 #(.JTAG_CHAIN(1)) bscan(
    .CAPTURE(scan_capture),.DRCK(scan_drck),.RESET(scan_reset),.RUNTEST(),
    .SEL(scan_sel),.SHIFT(scan_shift),.TCK(),.TDI(scan_tdi),
    .TDO(scan_tdo),.UPDATE(scan_update));
`endif
  reg [7:0] rx_shift=0,tx_shift=0;
  reg [2:0] bit_count=0;
  reg tx_loaded=0;
  wire rx_byte=scan_sel&&scan_shift&&(bit_count==3'd7);
  wire [7:0] completed_rx={scan_tdi,rx_shift[7:1]};
  wire rx_fifo_ready;
  wire tx_fifo_valid;
  wire [7:0] tx_fifo_data;
  wire tx_pop=tx_fifo_valid&&((scan_capture&&!tx_loaded)||
      (scan_sel&&scan_shift&&(bit_count==3'd7)));
  assign scan_tdo=tx_loaded?tx_shift[0]:1'b0;

  async_fifo #(.WIDTH(8),.ADDR_BITS(4)) rx_fifo(
    .wclk(scan_drck),.wrst(scan_reset),.wvalid(rx_byte),
    .wready(rx_fifo_ready),.wdata(completed_rx),
    .rclk(sys_clk),.rrst(sys_rst),.rvalid(rx_valid),
    .rready(rx_ready),.rdata(rx_data));
  async_fifo #(.WIDTH(8),.ADDR_BITS(4)) tx_fifo(
    .wclk(sys_clk),.wrst(sys_rst),.wvalid(tx_valid),
    .wready(tx_ready),.wdata(tx_data),
    .rclk(scan_drck),.rrst(scan_reset),.rvalid(tx_fifo_valid),
    .rready(tx_pop),.rdata(tx_fifo_data));

  always @(posedge scan_drck)begin
    if(scan_reset)begin rx_shift<=0;tx_shift<=0;bit_count<=0;tx_loaded<=0;end
    else if(scan_capture)begin
      bit_count<=0;
      if(!tx_loaded)begin
        tx_shift<=tx_fifo_valid?tx_fifo_data:8'h00;tx_loaded<=tx_fifo_valid;
      end
    end else if(scan_sel&&scan_shift)begin
      rx_shift<={scan_tdi,rx_shift[7:1]};
      if(bit_count==3'd7)begin
        bit_count<=0;
        tx_shift<=tx_fifo_valid?tx_fifo_data:8'h00;tx_loaded<=tx_fifo_valid;
      end else begin bit_count<=bit_count+1'b1;tx_shift<=tx_shift>>1;end
    end
  end
endmodule
