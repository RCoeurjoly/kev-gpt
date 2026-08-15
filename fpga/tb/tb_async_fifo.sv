`timescale 1ns/1ps
module tb_async_fifo;
  reg wclk=0,rclk=0,wrst=1,rrst=1,wvalid=0,rready=0;
  reg [7:0] wdata=0;wire wready,rvalid;wire [7:0] rdata;
  integer sent=0,received=0,cycles=0;
  always #3.5 wclk=~wclk;
  always #5.5 rclk=~rclk;
  async_fifo #(.WIDTH(8),.ADDR_BITS(3)) dut(
    .wclk(wclk),.wrst(wrst),.wvalid(wvalid),.wready(wready),.wdata(wdata),
    .rclk(rclk),.rrst(rrst),.rvalid(rvalid),.rready(rready),.rdata(rdata));
  always @(posedge wclk) begin
    if(wrst)begin wvalid<=0;wdata<=0;sent<=0;end
    else begin
      wvalid<=(sent<200);
      if(wvalid&&wready)begin sent<=sent+1;wdata<=sent[7:0]+1'b1;end
    end
  end
  always @(posedge rclk) begin
    cycles<=cycles+1;
    if(rrst)begin rready<=0;received<=0;end
    else begin
      rready<=((cycles%5)!=0);
      if(rvalid&&rready)begin
        if((^rdata)===1'bx||rdata!==received[7:0])begin
          $display("ASYNC_FIFO_FAIL got=%0d expected=%0d",rdata,received);$fatal(1);
        end
        received<=received+1;
        if(received==199)begin $display("ASYNC_FIFO_PASS count=200");$finish;end
      end
    end
  end
  initial begin
    #41 wrst=0;#26 rrst=0;
    #20000 $display("ASYNC_FIFO_TIMEOUT sent=%0d received=%0d",sent,received);$fatal(1);
  end
endmodule
