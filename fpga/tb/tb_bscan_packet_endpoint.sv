`timescale 1ns/1ps
module tb_bscan_packet_endpoint;
  reg sys_clk=0,sys_rst=1;
  reg scan_drck=0,scan_reset=1,scan_sel=0,scan_shift=0,scan_capture=0;
  reg scan_update=0,scan_tdi=0;wire scan_tdo;
  wire rx_valid,tx_ready;reg rx_ready,tx_valid;
  wire [7:0] rx_data;reg [7:0] tx_data=0;
  integer echoed=0,collected=0,i;reg [7:0] got;
  always #5 sys_clk=~sys_clk;
  bscan_packet_endpoint dut(
    .sys_clk(sys_clk),.sys_rst(sys_rst),.rx_valid(rx_valid),.rx_ready(rx_ready),
    .rx_data(rx_data),.tx_valid(tx_valid),.tx_ready(tx_ready),.tx_data(tx_data),
    .scan_drck(scan_drck),.scan_reset(scan_reset),.scan_sel(scan_sel),
    .scan_shift(scan_shift),.scan_capture(scan_capture),.scan_update(scan_update),
    .scan_tdi(scan_tdi),.scan_tdo(scan_tdo));
  always @* begin rx_ready=tx_ready;tx_valid=rx_valid;tx_data=rx_data^8'ha5;end
  always @(posedge sys_clk)if(rx_valid&&rx_ready)echoed<=echoed+1;
  task pulse_drck;begin #3 scan_drck=1;#3 scan_drck=0;end endtask
  task shift_byte;input [7:0] value;output [7:0] observed;integer bitno;begin
    observed=0;
    for(bitno=0;bitno<8;bitno=bitno+1)begin
      scan_tdi=value[bitno];#1;observed[bitno]=scan_tdo;pulse_drck;
    end
  end endtask
  initial begin
    #31 sys_rst=0;scan_reset=0;scan_sel=1;scan_shift=1;
    for(i=0;i<64;i=i+1)begin
      shift_byte(i[7:0],got);
      if(got!=0)begin
        if(got!==(collected[7:0]^8'ha5))begin
          $display("BSCAN_ENDPOINT_FAIL index=%0d got=%02x",collected,got);$fatal(1);
        end
        collected=collected+1;
      end
    end
    wait(echoed==64);#100;scan_shift=0;pulse_drck;pulse_drck;pulse_drck;scan_shift=1;
    scan_capture=1;pulse_drck;scan_capture=0;
    while(collected<64)begin
      shift_byte(0,got);
      if(got!=0)begin
        if(got!==(collected[7:0]^8'ha5))begin
          $display("BSCAN_ENDPOINT_FAIL index=%0d got=%02x",collected,got);$fatal(1);
        end
        collected=collected+1;
      end
    end
    $display("BSCAN_ENDPOINT_PASS bytes=64");$finish;
  end
  initial begin #50000;$display("BSCAN_ENDPOINT_TIMEOUT echoed=%0d",echoed);$fatal(1);end
endmodule
