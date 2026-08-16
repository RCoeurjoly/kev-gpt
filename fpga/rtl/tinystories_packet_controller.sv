module tinystories_packet_controller(
  input wire clk,input wire rst,
  input wire rx_valid,output wire rx_ready,input wire [7:0] rx_data,
  output wire tx_valid,input wire tx_ready,output wire [7:0] tx_data,
  output wire seq_clear,output wire prompt_valid,input wire prompt_ready,
  output wire [15:0] prompt_token,output wire seq_start,
  output wire [5:0] requested_tokens,
  input wire seq_token_valid,output wire seq_token_ready,
  input wire [15:0] seq_token_id,input wire seq_busy,input wire seq_error,
  input wire [7:0] seq_error_code,input wire [62:0] seq_debug_status,
  input wire [95:0] seq_debug_embedding,input wire [95:0] seq_debug_layernorm
);
  localparam RX=4'd0,CLEAR=4'd1,LOAD=4'd2,START=4'd3,RUN=4'd4,
             BUILD=4'd5,CRC_WRITE=4'd6,SEND=4'd7;
  localparam STATUS_OK=8'd0,STATUS_BAD_HEADER=8'd1,STATUS_BAD_CRC=8'd2,
             STATUS_CONTEXT=8'd3,STATUS_ACCELERATOR=8'd16;
  reg [3:0] state=RX;
  reg [7:0] packet[0:73],reply[0:80];
  reg [15:0] outputs[0:31];
  reg [6:0] rx_index=0,expected_length=0,build_index=0,payload_length=0;
  reg [6:0] send_index=0,reply_length=0;
  reg [5:0] prompt_count=0,generation_count=0,load_index=0,output_count=0;
  reg [1:0] crc_index=0;
  reg [7:0] reply_status=0;
  reg [31:0] request_crc=32'hffffffff,reply_crc=32'hffffffff,final_crc=0;
  reg [63:0] cycle_count=0;
  reg saw_busy=0;
  reg diagnostic_sending=0;reg [6:0] diagnostic_index=0;

  assign rx_ready=(state==RX)||(state==RUN&&!diagnostic_sending);
  assign tx_valid=diagnostic_sending||(state==SEND);
  function automatic [7:0] diagnostic_byte;
    input [6:0] which;begin case(which)
      0:diagnostic_byte=8'h44;1:diagnostic_byte=8'h42;
      2:diagnostic_byte=8'h47;3:diagnostic_byte=8'h31;
      4:diagnostic_byte=cycle_count[7:0];5:diagnostic_byte=cycle_count[15:8];
      6:diagnostic_byte=cycle_count[23:16];7:diagnostic_byte=cycle_count[31:24];
      8:diagnostic_byte={1'b0,seq_debug_status[6:0]};
      9:diagnostic_byte={5'b0,seq_debug_status[9:7]};
      10:diagnostic_byte={2'b0,seq_debug_status[15:10]};
      11:diagnostic_byte=seq_debug_status[23:16];
      12:diagnostic_byte={7'b0,seq_debug_status[24]};
      13:diagnostic_byte={5'b0,seq_debug_status[27:25]};
      14:diagnostic_byte=seq_debug_status[35:28];
      15:diagnostic_byte=seq_debug_status[43:36];
      16:diagnostic_byte={5'b0,seq_debug_status[46:44]};
      17:diagnostic_byte={2'b0,seq_debug_status[52:47]};
      18:diagnostic_byte={2'b0,seq_debug_status[58:53]};
      19:diagnostic_byte={3'b0,seq_error,seq_debug_status[62:59]};
      20:diagnostic_byte=seq_debug_embedding[7:0];
      21:diagnostic_byte=seq_debug_embedding[15:8];
      22:diagnostic_byte=seq_debug_embedding[23:16];
      23:diagnostic_byte=seq_debug_embedding[31:24];
      24:diagnostic_byte=seq_debug_embedding[39:32];
      25:diagnostic_byte=seq_debug_embedding[47:40];
      26:diagnostic_byte=seq_debug_embedding[55:48];
      27:diagnostic_byte=seq_debug_embedding[63:56];
      28:diagnostic_byte=seq_debug_embedding[71:64];
      29:diagnostic_byte=seq_debug_embedding[79:72];
      30:diagnostic_byte=seq_debug_embedding[87:80];
      31:diagnostic_byte=seq_debug_embedding[95:88];
      32:diagnostic_byte=seq_debug_layernorm[7:0];
      33:diagnostic_byte=seq_debug_layernorm[15:8];
      34:diagnostic_byte=seq_debug_layernorm[23:16];
      35:diagnostic_byte=seq_debug_layernorm[31:24];
      36:diagnostic_byte=seq_debug_layernorm[39:32];
      37:diagnostic_byte=seq_debug_layernorm[47:40];
      38:diagnostic_byte=seq_debug_layernorm[55:48];
      39:diagnostic_byte=seq_debug_layernorm[63:56];
      40:diagnostic_byte=seq_debug_layernorm[71:64];
      41:diagnostic_byte=seq_debug_layernorm[79:72];
      42:diagnostic_byte=seq_debug_layernorm[87:80];
      default:diagnostic_byte=seq_debug_layernorm[95:88];
    endcase end
  endfunction
  assign tx_data=diagnostic_sending?diagnostic_byte(diagnostic_index):reply[send_index];
  assign seq_clear=(state==CLEAR);
  assign prompt_valid=(state==LOAD);
  assign prompt_token={packet[7+(load_index<<1)],packet[6+(load_index<<1)]};
  assign seq_start=(state==START);
  assign requested_tokens=generation_count;
  assign seq_token_ready=(state==RUN);

  function automatic [31:0] crc_byte;
    input [31:0] old_crc;input [7:0] byte_value;
    integer bit_number;reg [31:0] value;begin
      value=old_crc^byte_value;
      for(bit_number=0;bit_number<8;bit_number=bit_number+1)
        value=value[0]?(value>>1)^32'hedb88320:value>>1;
      crc_byte=value;
    end
  endfunction

  function automatic [7:0] make_reply_byte;
    input [6:0] which;integer token_number,shift_amount;begin
      if(which==0)make_reply_byte=8'h52;
      else if(which==1)make_reply_byte=8'h4b;
      else if(which==2)make_reply_byte=8'd1;
      else if(which==3)make_reply_byte=reply_status;
      else if(which==4)make_reply_byte=output_count;
      else if(which<13)begin
        shift_amount=(which-5)*8;make_reply_byte=cycle_count>>shift_amount;
      end else begin
        token_number=(which-13)>>1;
        make_reply_byte=(which[0])?outputs[token_number][7:0]:outputs[token_number][15:8];
      end
    end
  endfunction

  task automatic begin_reply;
    input [7:0] status;begin
      reply_status<=status;build_index<=0;reply_crc<=32'hffffffff;
      payload_length<=13+(output_count<<1);state<=BUILD;
    end
  endtask

  always @(posedge clk)begin
    if(rst)begin
      state<=RX;rx_index<=0;expected_length<=0;request_crc<=32'hffffffff;
      output_count<=0;cycle_count<=0;saw_busy<=0;diagnostic_sending<=0;
    end else begin
      if(diagnostic_sending&&tx_ready)begin
        if(diagnostic_index==43)diagnostic_sending<=0;
        else diagnostic_index<=diagnostic_index+1'b1;
      end
      case(state)
      RX:if(rx_valid)begin
        if(rx_index==0)begin
          if(rx_data==8'h47)begin packet[0]<=rx_data;rx_index<=1;
            request_crc<=crc_byte(32'hffffffff,rx_data);output_count<=0;end
        end else if(rx_index==1)begin
          if(rx_data==8'h4b)begin packet[1]<=rx_data;rx_index<=2;
            request_crc<=crc_byte(request_crc,rx_data);end
          else if(rx_data==8'h47)begin packet[0]<=rx_data;rx_index<=1;
            request_crc<=crc_byte(32'hffffffff,rx_data);end
          else begin rx_index<=0;request_crc<=32'hffffffff;end
        end else begin
          packet[rx_index]<=rx_data;
          if(rx_index==4)expected_length<=(rx_data<=32?10+(rx_data<<1):10);
          if(rx_index<5||(expected_length!=0&&rx_index<expected_length-4))
            request_crc<=crc_byte(request_crc,rx_data);
          if(expected_length!=0&&rx_index==expected_length-1)begin
            rx_index<=0;expected_length<=0;
            prompt_count<=packet[4];generation_count<=packet[5];output_count<=0;
            cycle_count<=0;
            if(packet[2]!=1||packet[3]!=1||packet[4]==0)
              begin_reply(STATUS_BAD_HEADER);
            else if(packet[4]>32||({1'b0,packet[4]}+{1'b0,packet[5]}>32))
              begin_reply(STATUS_CONTEXT);
            else if({rx_data,packet[expected_length-2],packet[expected_length-3],
                     packet[expected_length-4]}!=(request_crc^32'hffffffff))
              begin_reply(STATUS_BAD_CRC);
            else begin load_index<=0;state<=CLEAR;end
          end else rx_index<=rx_index+1'b1;
        end
      end
      CLEAR:begin
        if(generation_count==0)begin output_count<=0;begin_reply(STATUS_OK);end
        else state<=LOAD;
      end
      LOAD:if(prompt_ready)begin
        if(load_index+1>=prompt_count)state<=START;
        else load_index<=load_index+1'b1;
      end
      START:begin cycle_count<=0;saw_busy<=0;state<=RUN;end
      RUN:begin
        cycle_count<=cycle_count+1'b1;
        if(rx_valid&&rx_data==8'h3f&&!diagnostic_sending)begin
          diagnostic_index<=0;diagnostic_sending<=1;
        end
        if(seq_busy)saw_busy<=1;
        if(seq_token_valid&&seq_token_ready)begin
          outputs[output_count]<=seq_token_id;output_count<=output_count+1'b1;
        end
        if(seq_error)begin output_count<=0;begin_reply(STATUS_ACCELERATOR+seq_error_code);end
        else if(saw_busy&&!seq_busy)begin_reply(STATUS_OK);
      end
      BUILD:begin
        reply[build_index]<=make_reply_byte(build_index);
        reply_crc<=crc_byte(reply_crc,make_reply_byte(build_index));
        if(build_index+1>=payload_length)begin
          final_crc<=crc_byte(reply_crc,make_reply_byte(build_index))^32'hffffffff;
          crc_index<=0;state<=CRC_WRITE;
        end else build_index<=build_index+1'b1;
      end
      CRC_WRITE:begin
        reply[payload_length+crc_index]<=final_crc>>(crc_index*8);
        if(crc_index==3)begin reply_length<=payload_length+4;send_index<=0;state<=SEND;end
        else crc_index<=crc_index+1'b1;
      end
      SEND:if(tx_valid&&tx_ready)begin
        if(send_index+1>=reply_length)begin
          state<=RX;rx_index<=0;expected_length<=0;request_crc<=32'hffffffff;
        end else send_index<=send_index+1'b1;
      end
      default:state<=RX;
      endcase
    end
  end
endmodule
