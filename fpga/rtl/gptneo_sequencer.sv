`timescale 1ns / 1ps

// Package-driven, batch-one GPT-Neo decoder.  One arithmetic datapath is
// reused across all eight layers and all sequence positions so the complete
// pretrained model remains resident in the XC7K480T block RAM budget.
module gptneo_sequencer #(
    parameter CHECK_PACKAGE_TAG=1,
    parameter integer TMAX=32,
    parameter integer D=64,
    parameter integer HIDDEN=256,
    parameter integer VOCAB=50257,
    parameter integer WDEPTH_BYTES=3825235,
    parameter MODEL_FILE="model_image.mem",
    parameter GELU_FILE="gptneo_gelu.mem",
    parameter EXP_FILE="gptneo_exp.mem"
) (
    input wire clk,input wire rst,input wire clear,
    input wire prompt_valid,output wire prompt_ready,input wire [15:0] prompt_token,
    input wire start,input wire [5:0] requested_tokens,input wire [31:0] package_tag,
    output reg token_valid,input wire token_ready,output reg [15:0] token_id,
    output wire busy,output reg error,output reg [7:0] error_code,
    output wire [62:0] debug_status,output reg [95:0] debug_embedding,
    output wire [95:0] debug_layernorm
);
`include "gptneo_package.svh"
    localparam ERR_PACKAGE_HASH=8'd1,ERR_CONTEXT=8'd2;
    localparam integer LAYER_BYTES=51712,LAYER_SCALE_BYTES=1728,LAYER_ACT_BYTES=3456;

    reg [15:0] tokens[0:TMAX-1];
    reg signed [31:0] xmem[0:TMAX*D-1],norm[0:D-1];
    reg signed [31:0] qbuf[0:D-1],kbuf[0:D-1],vbuf[0:D-1],ctxbuf[0:D-1];
    reg signed [31:0] hidden[0:HIDDEN-1];
    reg [5:0] token_count,generated_count,target_count;
    reg [2:0] layer;reg [5:0] position;reg [8:0] index;
    reg [8:0] vector_index;reg [2:0] gemv_op;
    reg signed [31:0] token_component;reg signed [7:0] token_code,position_code;
    reg signed [31:0] best_logit;reg [15:0] best_token;

    localparam OP_Q=3'd0,OP_K=3'd1,OP_V=3'd2,OP_OUT=3'd3,
               OP_FC=3'd4,OP_PROJ=3'd5,OP_LM=3'd6;
    localparam IDLE=7'd0,EMB_CODE_REQ=7'd1,EMB_CODE_WAIT=7'd2,
      EMB_SCALE_REQ=7'd3,EMB_SCALE_WAIT=7'd4,POS_CODE_REQ=7'd5,
      POS_CODE_WAIT=7'd6,POS_SCALE_REQ=7'd7,POS_SCALE_WAIT=7'd8,
      LN_G_REQ=7'd9,LN_G_WAIT=7'd10,LN_B_REQ=7'd11,LN_B_WAIT=7'd12,
      LN_FEED=7'd13,LN_START=7'd14,LN_COLLECT=7'd15,
      GEMV_LOAD=7'd16,GEMV_START=7'd17,GEMV_COLLECT=7'd18,
      ATTN_START=7'd19,ATTN_FEED=7'd20,ATTN_COLLECT=7'd21,
      GELU_FEED=7'd22,GELU_COLLECT=7'd23,NEXT_POSITION=7'd24,
      FINAL_LN_SETUP=7'd25,LM_SETUP=7'd26,EMIT=7'd27,ERROR_HOLD=7'd28;
    reg [6:0] state,ln_return_state;
    reg ln_is_second,ln_is_final;

    wire gemv_busy,gemv_activation_ready,gemv_out_valid,gemv_raw_valid;
    reg gemv_activation_valid,gemv_start,gemv_raw_read,gemv_raw_ready;
    reg signed [31:0] gemv_activation_data_q16;
    reg [$clog2(WDEPTH_BYTES)-1:0] gemv_raw_addr;
    wire [31:0] gemv_raw_data;
    wire [15:0] gemv_out_index;wire signed [31:0] gemv_out_value;
    wire signed [63:0] gemv_out_accumulator;wire signed [7:0] gemv_out_code;

    reg ln_in_valid,ln_start;wire ln_in_ready,ln_out_valid,ln_busy;
    reg signed [31:0] ln_x,ln_gamma,ln_beta;wire [5:0] ln_out_index;
    wire signed [31:0] ln_y;
    reg attn_start,attn_cache_reset,attn_in_valid;wire attn_in_ready,attn_out_valid,attn_busy;
    reg signed [31:0] attn_q,attn_k,attn_v;wire [5:0] attn_out_index;
    wire signed [31:0] attn_out_context;
    reg gelu_in_valid;wire gelu_in_ready,gelu_out_valid;reg signed [15:0] gelu_in_data;
    wire signed [15:0] gelu_out_data;wire [1:0] gelu_debug_state;

    wire [21:0] layer_base=layer*LAYER_BYTES;
    wire [21:0] layer_scale=GPTNEO_SCALE_BASE+layer*LAYER_SCALE_BYTES;
    wire [21:0] layer_act=GPTNEO_ACTIVATION_SCALE_BASE+192+layer*LAYER_ACT_BYTES;
    wire [21:0] token_scale=GPTNEO_SCALE_BASE+13920;
    wire [21:0] position_scale=GPTNEO_SCALE_BASE+13824;

    reg [21:0] matrix_base,input_scale_base,weight_scale_base,output_scale_base,bias_base;
    reg [15:0] m_count;reg [8:0] k_count,row_stride;reg has_output_scale,has_bias;
    always @* begin
        matrix_base=0;input_scale_base=0;weight_scale_base=0;output_scale_base=0;
        bias_base=0;m_count=64;k_count=64;row_stride=64;has_output_scale=1;has_bias=0;
        case(gemv_op)
          OP_Q:begin matrix_base=layer_base+8448;weight_scale_base=layer_scale+384;
            input_scale_base=layer_act+768;output_scale_base=layer_act+960;end
          OP_K:begin matrix_base=layer_base;weight_scale_base=layer_scale;
            input_scale_base=layer_act;output_scale_base=layer_act+192;end
          OP_V:begin matrix_base=layer_base+12544;weight_scale_base=layer_scale+576;
            input_scale_base=layer_act+1152;output_scale_base=layer_act+1344;end
          OP_OUT:begin matrix_base=layer_base+4352;weight_scale_base=layer_scale+192;
            input_scale_base=layer_act+384;output_scale_base=layer_act+576;
            bias_base=layer_base+4096;has_bias=1;end
          OP_FC:begin matrix_base=layer_base+18688;weight_scale_base=layer_scale+768;
            input_scale_base=layer_act+1536;output_scale_base=layer_act+1728;
            bias_base=layer_base+17664;has_bias=1;m_count=256;end
          OP_PROJ:begin matrix_base=layer_base+35328;weight_scale_base=layer_scale+1536;
            input_scale_base=layer_act+2496;output_scale_base=layer_act+3264;
            bias_base=layer_base+35072;has_bias=1;k_count=256;row_stride=256;end
          default:begin matrix_base=GPTNEO_TENSOR_TOKEN_EMBEDDING_WEIGHT_OFFSET;
            weight_scale_base=token_scale;input_scale_base=GPTNEO_ACTIVATION_SCALE_BASE;
            m_count=VOCAB;has_output_scale=0;end
        endcase
    end

    gptneo_resident_gemv #(.MMAX(VOCAB),.KMAX(HIDDEN),.WDEPTH_BYTES(WDEPTH_BYTES),
      .WEIGHT_FILE(MODEL_FILE)) gemv(
      .clk(clk),.rst(rst),.weight_we(1'b0),.weight_addr(0),.weight_data(0),
      .raw_read(gemv_raw_read),.raw_addr(gemv_raw_addr),.raw_valid(gemv_raw_valid),
      .raw_ready(gemv_raw_ready),.raw_data(gemv_raw_data),
      .activation_valid(gemv_activation_valid),.activation_ready(gemv_activation_ready),
      .activation_data_q16(gemv_activation_data_q16),.start(gemv_start),.matrix_id(gemv_op),
      .matrix_base(matrix_base),.row_stride(row_stride),.input_scale_base(input_scale_base),
      .weight_scale_base(weight_scale_base),.output_scale_base(output_scale_base),
      .bias_base(bias_base),.has_output_scale(has_output_scale),.has_bias(has_bias),
      .m_count(m_count),.k_count(k_count),.out_valid(gemv_out_valid),.out_ready(1'b1),
      .out_index(gemv_out_index),.out_accumulator(gemv_out_accumulator),
      .out_value_q16(gemv_out_value),.out_code(gemv_out_code),.busy(gemv_busy));
    gptneo_layernorm #(.D(D)) layernorm(.clk(clk),.rst(rst),.in_valid(ln_in_valid),
      .in_ready(ln_in_ready),.in_x(ln_x),.in_gamma(ln_gamma),.in_beta(ln_beta),
      .start(ln_start),.out_valid(ln_out_valid),.out_ready(1'b1),.out_index(ln_out_index),
      .out_y(ln_y),.busy(ln_busy),.debug_status(debug_layernorm));
    gptneo_attention #(.D(D),.TMAX(TMAX),.EXP_FILE(EXP_FILE)) attention(.clk(clk),.rst(rst),
      .cache_reset(attn_cache_reset),.start(attn_start),.position(position),
      .in_valid(attn_in_valid),.in_ready(attn_in_ready),.in_q(attn_q),.in_k(attn_k),
      .in_v(attn_v),.out_valid(attn_out_valid),.out_ready(1'b1),
      .out_index(attn_out_index),.out_context(attn_out_context),.busy(attn_busy));
    gptneo_gelu #(.LUT_FILE(GELU_FILE)) gelu(.clk(clk),.rst(rst),.in_valid(gelu_in_valid),
      .in_ready(gelu_in_ready),.in_data(gelu_in_data),.out_valid(gelu_out_valid),
      .out_ready(1'b1),.out_data(gelu_out_data),.debug_state(gelu_debug_state));

    assign prompt_ready=(state==IDLE)&&!start&&(token_count<TMAX);
    assign busy=(state!=IDLE);
    // LSB-first USER2 diagnostic layout: state, layer, position, index,
    // GEMV operation/output index, sub-engine busy flags and token counters.
    assign debug_status={gelu_out_valid,gelu_in_ready,gelu_debug_state,
      generated_count,token_count,attn_busy,ln_busy,
      gemv_busy,gemv_out_index,gemv_op,index,position,layer,state};
    function automatic signed [31:0] round_scale;
      input signed [7:0] code;input [23:0] scale;reg signed [39:0] product,mag;begin
        product=code*$signed({1'b0,scale});mag=product<0?-product:product;
        mag=(mag+128)>>>8;round_scale=product<0?-mag:mag;end
    endfunction
    function automatic signed [15:0] q16_to_q12;
      input signed [31:0] value;reg signed [31:0] mag,r;begin
        mag=value<0?-value:value;r=(mag+8)>>>4;if(value<0)r=-r;
        if(r>32767)q16_to_q12=32767;else if(r< -32768)q16_to_q12=-32768;
        else q16_to_q12=r[15:0];end
    endfunction

    always @(posedge clk) begin
      gemv_activation_valid<=0;gemv_start<=0;gemv_raw_read<=0;gemv_raw_ready<=0;
      ln_in_valid<=0;ln_start<=0;attn_start<=0;attn_cache_reset<=0;attn_in_valid<=0;
      gelu_in_valid<=0;
      if(rst||clear)begin state<=IDLE;token_count<=0;generated_count<=0;token_valid<=0;
        error<=0;error_code<=0;index<=0;position<=0;layer<=0;
        debug_embedding<=0;end
      else begin
        if(prompt_valid&&prompt_ready)begin tokens[token_count]<=prompt_token;token_count<=token_count+1'b1;end
        case(state)
          IDLE:if(start)begin
            token_valid<=0;error<=0;generated_count<=0;target_count<=requested_tokens;
            if(CHECK_PACKAGE_TAG&&package_tag!=GPTNEO_PACKAGE_TAG)begin error<=1;error_code<=ERR_PACKAGE_HASH;state<=ERROR_HOLD;end
            else if(token_count==0||token_count+requested_tokens>TMAX)begin error<=1;error_code<=ERR_CONTEXT;state<=ERROR_HOLD;end
            else begin position<=0;index<=0;state<=EMB_CODE_REQ;end
          end
          EMB_CODE_REQ:if(!gemv_busy)begin
            gemv_raw_addr<=GPTNEO_TENSOR_TOKEN_EMBEDDING_WEIGHT_OFFSET+tokens[position]*D+index;
            gemv_raw_read<=1;state<=EMB_CODE_WAIT;end
          EMB_CODE_WAIT:if(gemv_raw_valid)begin token_code<=gemv_raw_data[7:0];
            gemv_raw_ready<=1;state<=EMB_SCALE_REQ;end
          EMB_SCALE_REQ:if(!gemv_busy)begin gemv_raw_addr<=token_scale+tokens[position]*3;
            gemv_raw_read<=1;state<=EMB_SCALE_WAIT;end
          EMB_SCALE_WAIT:if(gemv_raw_valid)begin
            token_component<=round_scale(token_code,gemv_raw_data[23:0]);
            gemv_raw_ready<=1;state<=POS_CODE_REQ;end
          POS_CODE_REQ:if(!gemv_busy)begin gemv_raw_addr<=GPTNEO_TENSOR_POSITION_EMBEDDING_WEIGHT_OFFSET+position*D+index;
            gemv_raw_read<=1;state<=POS_CODE_WAIT;end
          POS_CODE_WAIT:if(gemv_raw_valid)begin position_code<=gemv_raw_data[7:0];gemv_raw_ready<=1;state<=POS_SCALE_REQ;end
          POS_SCALE_REQ:if(!gemv_busy)begin gemv_raw_addr<=position_scale+position*3;
            gemv_raw_read<=1;state<=POS_SCALE_WAIT;end
          POS_SCALE_WAIT:if(gemv_raw_valid)begin
            xmem[position*D+index]<=token_component+round_scale(position_code,gemv_raw_data[23:0]);
            if(position==0&&index==0)debug_embedding<=
              {position_code,gemv_raw_data[23:0],token_component,
               token_component+round_scale(position_code,gemv_raw_data[23:0])};
            gemv_raw_ready<=1;
            if(index==D-1)begin index<=0;if(position==token_count-1)begin position<=0;layer<=0;
              attn_cache_reset<=1;ln_is_second<=0;ln_is_final<=0;state<=LN_G_REQ;end
              else begin position<=position+1'b1;state<=EMB_CODE_REQ;end end
            else begin index<=index+1'b1;state<=EMB_CODE_REQ;end
          end
          LN_G_REQ:if(!gemv_busy)begin
            if(ln_is_final)gemv_raw_addr<=GPTNEO_TENSOR_FINAL_LN_WEIGHT_OFFSET+index*4;
            else if(ln_is_second)gemv_raw_addr<=layer_base+17408+index*4;
            else gemv_raw_addr<=layer_base+16896+index*4;
            gemv_raw_read<=1;state<=LN_G_WAIT;end
          LN_G_WAIT:if(gemv_raw_valid)begin ln_gamma<=gemv_raw_data;gemv_raw_ready<=1;state<=LN_B_REQ;end
          LN_B_REQ:if(!gemv_busy)begin
            if(ln_is_final)gemv_raw_addr<=GPTNEO_TENSOR_FINAL_LN_BIAS_OFFSET+index*4;
            else if(ln_is_second)gemv_raw_addr<=layer_base+17152+index*4;
            else gemv_raw_addr<=layer_base+16640+index*4;
            gemv_raw_read<=1;state<=LN_B_WAIT;end
          LN_B_WAIT:if(gemv_raw_valid)begin ln_beta<=gemv_raw_data;gemv_raw_ready<=1;state<=LN_FEED;end
          LN_FEED:if(ln_in_ready)begin ln_x<=xmem[position*D+index];ln_in_valid<=1;
            if(index==D-1)begin index<=0;state<=LN_START;end else begin index<=index+1'b1;state<=LN_G_REQ;end end
          LN_START:begin ln_start<=1;state<=LN_COLLECT;end
          LN_COLLECT:if(ln_out_valid)begin norm[ln_out_index]<=ln_y;
            if(ln_out_index==D-1)begin vector_index<=0;if(ln_is_final)begin gemv_op<=OP_LM;state<=GEMV_LOAD;end
              else begin gemv_op<=ln_is_second?OP_FC:OP_Q;state<=GEMV_LOAD;end end
          end
          GEMV_LOAD:if(gemv_activation_ready)begin
            case(gemv_op) OP_OUT:gemv_activation_data_q16<=ctxbuf[vector_index];
              OP_PROJ:gemv_activation_data_q16<=hidden[vector_index];
              default:gemv_activation_data_q16<=norm[vector_index];endcase
            gemv_activation_valid<=1;
            if(vector_index==k_count-1)begin vector_index<=0;state<=GEMV_START;end
            else vector_index<=vector_index+1'b1;
          end
          GEMV_START:begin gemv_start<=1;if(gemv_op==OP_LM)begin best_logit<=-32'sh7fffffff;best_token<=0;end
            state<=GEMV_COLLECT;end
          GEMV_COLLECT:if(gemv_out_valid)begin
            case(gemv_op)
              OP_Q:qbuf[gemv_out_index]<=gemv_out_value;
              OP_K:kbuf[gemv_out_index]<=gemv_out_value;
              OP_V:vbuf[gemv_out_index]<=gemv_out_value;
              OP_OUT:xmem[position*D+gemv_out_index]<=xmem[position*D+gemv_out_index]+gemv_out_value;
              OP_FC:hidden[gemv_out_index]<=gemv_out_value;
              OP_PROJ:xmem[position*D+gemv_out_index]<=xmem[position*D+gemv_out_index]+gemv_out_value;
              default:if(gemv_out_value>best_logit)begin best_logit<=gemv_out_value;best_token<=gemv_out_index;end
            endcase
            if(gemv_out_index==m_count-1)begin
              case(gemv_op)
                OP_Q:begin gemv_op<=OP_K;vector_index<=0;state<=GEMV_LOAD;end
                OP_K:begin gemv_op<=OP_V;vector_index<=0;state<=GEMV_LOAD;end
                OP_V:begin index<=0;state<=ATTN_START;end
                OP_OUT:begin ln_is_second<=1;index<=0;state<=LN_G_REQ;end
                OP_FC:begin index<=0;state<=GELU_FEED;end
                OP_PROJ:state<=NEXT_POSITION;
                default:begin
                  if(gemv_out_value>best_logit)token_id<=gemv_out_index;else token_id<=best_token;
                  token_valid<=1;state<=EMIT;end
              endcase
            end
          end
          ATTN_START:begin attn_start<=1;state<=ATTN_FEED;end
          ATTN_FEED:if(attn_in_ready)begin attn_q<=qbuf[index];attn_k<=kbuf[index];attn_v<=vbuf[index];
            attn_in_valid<=1;if(index==D-1)begin index<=0;state<=ATTN_COLLECT;end else index<=index+1'b1;end
          ATTN_COLLECT:if(attn_out_valid)begin ctxbuf[attn_out_index]<=attn_out_context;
            if(attn_out_index==D-1)begin gemv_op<=OP_OUT;vector_index<=0;state<=GEMV_LOAD;end end
          GELU_FEED:if(gelu_in_ready)begin gelu_in_data<=q16_to_q12(hidden[index]);gelu_in_valid<=1;state<=GELU_COLLECT;end
          GELU_COLLECT:if(gelu_out_valid)begin hidden[index]<=$signed(gelu_out_data)<<<4;
            if(index==HIDDEN-1)begin gemv_op<=OP_PROJ;vector_index<=0;state<=GEMV_LOAD;end
            else begin index<=index+1'b1;state<=GELU_FEED;end end
          NEXT_POSITION:begin ln_is_second<=0;ln_is_final<=0;index<=0;
            if(position==token_count-1)begin position<=0;if(layer==7)begin position<=token_count-1;
              ln_is_final<=1;state<=LN_G_REQ;end else begin layer<=layer+1'b1;attn_cache_reset<=1;state<=LN_G_REQ;end end
            else begin position<=position+1'b1;state<=LN_G_REQ;end
          end
          EMIT:if(token_ready)begin token_valid<=0;tokens[token_count]<=token_id;token_count<=token_count+1'b1;
            generated_count<=generated_count+1'b1;
            if(generated_count+1>=target_count||token_id==16'd50256)state<=IDLE;
            else begin position<=0;index<=0;state<=EMB_CODE_REQ;end
          end
          ERROR_HOLD:if(clear)state<=IDLE;
          default:state<=IDLE;
        endcase
      end
    end
endmodule
