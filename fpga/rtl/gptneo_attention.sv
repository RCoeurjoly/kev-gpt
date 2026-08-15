`timescale 1ns / 1ps

// One GPT-Neo attention layer, D=64/NHEAD=16/HEAD_DIM=4, Q16.16 data.
// GPT-Neo's attention scores are intentionally unscaled. Softmax uses an
// exp(delta) Q0.20 LUT over [-16,0] with 1/256 spacing.
module gptneo_attention #(
    parameter integer D=64,
    parameter integer NHEAD=16,
    parameter integer HEAD_DIM=4,
    parameter integer TMAX=32,
    parameter EXP_FILE="gptneo_exp.mem"
) (
    input wire clk, input wire rst, input wire cache_reset,
    input wire start,
    input wire [$clog2(TMAX)-1:0] position,
    input wire in_valid, output wire in_ready,
    input wire signed [31:0] in_q, in_k, in_v,
    output reg out_valid, input wire out_ready,
    output reg [$clog2(D)-1:0] out_index,
    output reg signed [31:0] out_context,
    output wire busy
);
    (* ram_style="block" *) reg signed [31:0] kmem[0:TMAX*D-1];
    (* ram_style="block" *) reg signed [31:0] vmem[0:TMAX*D-1];
    reg signed [31:0] qmem[0:D-1];
    (* rom_style="block" *) reg [20:0] exp_lut[0:4095];
    initial $readmemh(EXP_FILE, exp_lut);
    reg signed [31:0] scoremem[0:TMAX-1];
    reg [20:0] expmem[0:TMAX-1];

    localparam IDLE=4'd0,LOAD=4'd1,SCORE_READ=4'd2,SCORE_MAC=4'd3,
               EXP_READ=4'd4,EXP_ACC=4'd5,CTX_READ=4'd6,CTX_ACC=4'd7,
               DIV_START=4'd8,DIV_WAIT=4'd9,EMIT=4'd10;
    reg [3:0] state;
    reg [$clog2(D)-1:0] load_index;
    reg [$clog2(NHEAD)-1:0] head;
    reg [$clog2(HEAD_DIM)-1:0] dimension;
    reg [$clog2(TMAX)-1:0] time_index;
    reg signed [95:0] accumulator;
    reg signed [95:0] product_total;
    reg signed [31:0] score_code, score_max;
    reg signed [31:0] delta_code;
    reg [12:0] exp_index;
    reg [63:0] exp_sum;
    reg [20:0] exp_value;
    reg signed [95:0] context_total;
    reg signed [95:0] rounded_context;
    reg signed [31:0] q_value, k_value, v_value;
    reg [20:0] probability_value;
    reg divider_start;
    reg signed [95:0] divider_numerator;
    wire divider_busy, divider_done;
    wire signed [31:0] divider_quotient;

    gptneo_iterative_divider context_divider(
        .clk(clk), .rst(rst || cache_reset), .start(divider_start),
        .numerator(divider_numerator), .denominator(exp_sum),
        .busy(divider_busy), .done(divider_done), .quotient(divider_quotient)
    );

    assign in_ready = (state == LOAD);
    assign busy = (state != IDLE);

    always @(posedge clk) begin
        if (rst || cache_reset) begin
            state<=IDLE; out_valid<=0; load_index<=0; divider_start<=0;
        end else begin
          divider_start <= 0;
          case(state)
            IDLE: if(start) begin load_index<=0; state<=LOAD; end
            LOAD: if(in_valid) begin
                qmem[load_index] <= in_q;
                kmem[position*D+load_index] <= in_k;
                vmem[position*D+load_index] <= in_v;
                if(load_index==D-1) begin
                    head<=0; dimension<=0; time_index<=0;
                    accumulator<=0; score_max<=-32'sh7fffffff; state<=SCORE_READ;
                end else load_index<=load_index+1'b1;
            end
            SCORE_READ: begin
                q_value <= qmem[head*HEAD_DIM+dimension];
                k_value <= kmem[time_index*D+head*HEAD_DIM+dimension];
                state <= SCORE_MAC;
            end
            SCORE_MAC: begin
                product_total = accumulator + $signed(q_value) * $signed(k_value);
                if(dimension==HEAD_DIM-1) begin
                    score_code = product_total >>> 24;
                    scoremem[time_index] <= score_code;
                    if(score_code > score_max) score_max <= score_code;
                    accumulator<=0; dimension<=0;
                    if(time_index==position) begin
                        time_index<=0; exp_sum<=0; state<=EXP_READ;
                    end else begin time_index<=time_index+1'b1; state<=SCORE_READ; end
                end else begin
                    accumulator<=product_total; dimension<=dimension+1'b1; state<=SCORE_READ;
                end
            end
            EXP_READ: begin
                delta_code = scoremem[time_index] - score_max;
                if(delta_code < -4096) exp_index = 0;
                else if(delta_code >= 0) exp_index = 4096;
                else exp_index = 4096 + delta_code;
                if(exp_index == 4096) exp_value <= 21'd1048576;
                else exp_value <= exp_lut[exp_index];
                state <= EXP_ACC;
            end
            EXP_ACC: begin
                expmem[time_index] <= exp_value;
                exp_sum <= exp_sum + exp_value;
                if(time_index==position) begin
                    time_index<=0; dimension<=0; accumulator<=0; state<=CTX_READ;
                end else begin time_index<=time_index+1'b1; state<=EXP_READ; end
            end
            CTX_READ: begin
                probability_value <= expmem[time_index];
                v_value <= vmem[time_index*D+head*HEAD_DIM+dimension];
                state <= CTX_ACC;
            end
            CTX_ACC: begin
                context_total = accumulator + $signed({1'b0, probability_value}) * $signed(v_value);
                if(time_index==position) begin
                    if(context_total < 0) rounded_context = context_total - $signed(exp_sum>>>1);
                    else rounded_context = context_total + $signed(exp_sum>>>1);
                    divider_numerator <= rounded_context;
                    state <= DIV_START;
                end else begin
                    accumulator<=context_total; time_index<=time_index+1'b1; state<=CTX_READ;
                end
            end
            DIV_START: begin
                divider_start <= 1;
                state <= DIV_WAIT;
            end
            DIV_WAIT: if(divider_done) begin
                out_context <= divider_quotient;
                out_index <= head*HEAD_DIM+dimension;
                out_valid <= 1;
                state <= EMIT;
            end
            EMIT: if(out_ready) begin
                out_valid<=0; accumulator<=0; time_index<=0;
                if(dimension==HEAD_DIM-1) begin
                    if(head==NHEAD-1) state<=IDLE;
                    else begin head<=head+1'b1; dimension<=0; score_max<=-32'sh7fffffff; state<=SCORE_READ; end
                end else begin dimension<=dimension+1'b1; state<=CTX_READ; end
            end
            default: state<=IDLE;
          endcase
        end
    end
endmodule
