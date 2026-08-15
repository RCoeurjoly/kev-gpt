module async_fifo #(
  parameter integer WIDTH=8,
  parameter integer ADDR_BITS=3
)(
  input wire wclk,input wire wrst,input wire wvalid,output wire wready,
  input wire [WIDTH-1:0] wdata,
  input wire rclk,input wire rrst,output wire rvalid,input wire rready,
  output wire [WIDTH-1:0] rdata
);
  localparam integer PTR_BITS=ADDR_BITS+1;
  reg [WIDTH-1:0] memory[0:(1<<ADDR_BITS)-1];
  reg [PTR_BITS-1:0] wbin=0,wgray=0,rbin=0,rgray=0;
  (* ASYNC_REG="TRUE" *) reg [PTR_BITS-1:0] rgray_w1=0,rgray_w2=0;
  (* ASYNC_REG="TRUE" *) reg [PTR_BITS-1:0] wgray_r1=0,wgray_r2=0;
  wire [PTR_BITS-1:0] wbin_candidate=wbin+1'b1;
  wire [PTR_BITS-1:0] wgray_candidate=(wbin_candidate>>1)^wbin_candidate;
  wire wpush=wvalid&&wready;
  wire rpop=rvalid&&rready;
  wire [PTR_BITS-1:0] wbin_next=wbin+wpush;
  wire [PTR_BITS-1:0] rbin_next=rbin+rpop;
  wire [PTR_BITS-1:0] wgray_next=(wbin_next>>1)^wbin_next;
  wire [PTR_BITS-1:0] rgray_next=(rbin_next>>1)^rbin_next;
  wire [PTR_BITS-1:0] full_compare={~rgray_w2[PTR_BITS-1:PTR_BITS-2],
                                            rgray_w2[PTR_BITS-3:0]};
  assign wready=(wgray_candidate!=full_compare);
  assign rvalid=(rgray!=wgray_r2);
  assign rdata=memory[rbin[ADDR_BITS-1:0]];

  always @(posedge wclk)begin
    if(wrst)begin wbin<=0;wgray<=0;rgray_w1<=0;rgray_w2<=0;end
    else begin
      rgray_w1<=rgray;rgray_w2<=rgray_w1;
      if(wpush)begin memory[wbin[ADDR_BITS-1:0]]<=wdata;wbin<=wbin_next;wgray<=wgray_next;end
    end
  end
  always @(posedge rclk)begin
    if(rrst)begin rbin<=0;rgray<=0;wgray_r1<=0;wgray_r2<=0;end
    else begin
      wgray_r1<=wgray;wgray_r2<=wgray_r1;
      if(rpop)begin rbin<=rbin_next;rgray<=rgray_next;end
    end
  end
endmodule
