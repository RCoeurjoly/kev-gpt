`timescale 1ns/1ps
module tb_tinystories_packet_controller;
  reg clk=0,rst=1,rx_valid=0,tx_ready=1,prompt_ready=1;
  reg [7:0] rx_data=0;wire rx_ready,tx_valid,prompt_valid,seq_clear,seq_start;
  wire [7:0] tx_data;wire [15:0] prompt_token;wire [5:0] requested_tokens;
  reg seq_token_valid=0,seq_busy=0,seq_error=0;reg [15:0] seq_token_id=0;
  reg [7:0] seq_error_code=0;wire seq_token_ready;
  reg [7:0] request[0:13],reply[0:31];integer i,reply_count=0,fake_state=0;
  reg [31:0] crc;
  always #5 clk=~clk;
  tinystories_packet_controller dut(
    .clk(clk),.rst(rst),.rx_valid(rx_valid),.rx_ready(rx_ready),.rx_data(rx_data),
    .tx_valid(tx_valid),.tx_ready(tx_ready),.tx_data(tx_data),
    .seq_clear(seq_clear),.prompt_valid(prompt_valid),.prompt_ready(prompt_ready),
    .prompt_token(prompt_token),.seq_start(seq_start),.requested_tokens(requested_tokens),
    .seq_token_valid(seq_token_valid),.seq_token_ready(seq_token_ready),
    .seq_token_id(seq_token_id),.seq_busy(seq_busy),.seq_error(seq_error),
    .seq_error_code(seq_error_code));
  function automatic [31:0] crc_byte;input [31:0] old;input [7:0] byte_value;
    integer bitno;reg [31:0] value;begin value=old^byte_value;
      for(bitno=0;bitno<8;bitno=bitno+1)
        value=value[0]?(value>>1)^32'hedb88320:value>>1;
      crc_byte=value;end
  endfunction
  always @(posedge clk)begin
    seq_token_valid<=0;
    if(seq_clear)begin fake_state<=0;seq_busy<=0;end
    if(seq_start)begin fake_state<=1;seq_busy<=1;end
    else case(fake_state)
      1:fake_state<=2;
      2:begin seq_token_id<=16'd11;seq_token_valid<=1;fake_state<=3;end
      3:if(seq_token_ready)begin seq_token_id<=16'd12;seq_token_valid<=1;fake_state<=4;end
      4:if(seq_token_ready)begin seq_busy<=0;fake_state<=5;end
    endcase
    if(tx_valid&&tx_ready)begin reply[reply_count]<=tx_data;reply_count<=reply_count+1;end
  end
  task send_request;begin
    for(i=0;i<14;i=i+1)begin
      while(!rx_ready)@(posedge clk);rx_data<=request[i];rx_valid<=1;
      @(posedge clk);rx_valid<=0;
    end
  end endtask
  initial begin
    request[0]=8'h47;request[1]=8'h4b;request[2]=1;request[3]=1;
    request[4]=2;request[5]=2;request[6]=1;request[7]=0;request[8]=2;
    request[9]=0;request[10]=8'hd3;request[11]=8'h15;request[12]=8'h74;request[13]=8'h14;
    #31 rst=0;send_request;wait(reply_count==21);@(posedge clk);
    if(reply[0]!==8'h52||reply[1]!==8'h4b||reply[2]!==1||reply[3]!==0||
       reply[4]!==2||reply[13]!==11||reply[14]!==0||reply[15]!==12||reply[16]!==0)begin
      $write("PACKET_CONTROLLER_FAIL malformed reply:");for(i=0;i<21;i=i+1)$write(" %02x",reply[i]);
      $display("");$fatal(1);end
    crc=32'hffffffff;for(i=0;i<17;i=i+1)crc=crc_byte(crc,reply[i]);crc=crc^32'hffffffff;
    if({reply[20],reply[19],reply[18],reply[17]}!==crc)begin
      $display("PACKET_CONTROLLER_FAIL crc");$fatal(1);end
    $display("PACKET_CONTROLLER_PASS tokens=2");$finish;
  end
  initial begin #100000;$display("PACKET_CONTROLLER_TIMEOUT replies=%0d",reply_count);$fatal(1);end
endmodule
