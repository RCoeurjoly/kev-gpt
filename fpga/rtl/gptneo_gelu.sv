`timescale 1ns / 1ps

// GPT-Neo gelu_new in signed Q4.12.  The 8192-entry table covers [-8, 8)
// at 1/512 spacing and uses the low three input bits for interpolation.
module gptneo_gelu #(
    parameter LUT_FILE = "gptneo_gelu.mem"
) (
    input  wire               clk,
    input  wire               rst,
    input  wire               in_valid,
    output wire               in_ready,
    input  wire signed [15:0] in_data,
    output reg                out_valid,
    input  wire               out_ready,
    output reg signed [15:0]  out_data
);
    (* rom_style = "block" *) reg signed [15:0] lut [0:8191];
    initial $readmemh(LUT_FILE, lut);

    localparam IDLE = 2'd0, READ = 2'd1, INTERP = 2'd2, HOLD = 2'd3;
    reg [1:0] state;
    reg [12:0] index;
    reg [2:0] fraction;
    reg signed [15:0] lower, upper;
    reg signed [16:0] difference;
    reg signed [19:0] step;
    wire [15:0] biased = in_data + 16'h8000;

    assign in_ready = (state == IDLE);

    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            out_valid <= 1'b0;
            out_data <= 16'sd0;
        end else begin
            case (state)
                IDLE: begin
                    out_valid <= 1'b0;
                    if (in_valid) begin
                        index <= biased[15:3];
                        fraction <= biased[2:0];
                        state <= READ;
                    end
                end
                READ: begin
                    lower <= lut[index];
                    upper <= lut[(index == 13'd8191) ? index : index + 1'b1];
                    state <= INTERP;
                end
                INTERP: begin
                    difference = $signed(upper) - $signed(lower);
                    step = difference * $signed({1'b0, fraction});
                    out_data <= $signed(lower) + (step >>> 3);
                    out_valid <= 1'b1;
                    state <= HOLD;
                end
                HOLD: begin
                    if (out_ready) begin
                        out_valid <= 1'b0;
                        state <= IDLE;
                    end
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
