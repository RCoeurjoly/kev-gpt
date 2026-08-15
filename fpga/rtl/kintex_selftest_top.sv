module kintex_selftest_top(
  input  logic       SYS_CLK,
  input  logic       SYS_RSTN,
  output logic [2:0] led_3bits_tri_o
);
  localparam logic [7:0] EXPECTED_SUM = 8'd136;

  logic [25:0] heartbeat_count;
  logic [4:0]  term;
  logic [7:0]  sum;
  logic        pass_latched;
  logic        fail_latched;

  always_ff @(posedge SYS_CLK or negedge SYS_RSTN) begin
    if (!SYS_RSTN) begin
      heartbeat_count <= 26'd0;
      term <= 5'd1;
      sum <= 8'd0;
      pass_latched <= 1'b0;
      fail_latched <= 1'b0;
    end else begin
      heartbeat_count <= heartbeat_count + 26'd1;

      if (!(pass_latched || fail_latched)) begin
        if (term == 5'd16) begin
          if (sum + term == EXPECTED_SUM) begin
            pass_latched <= 1'b1;
          end else begin
            fail_latched <= 1'b1;
          end
        end else begin
          sum <= sum + term;
          term <= term + 5'd1;
        end
      end
    end
  end

  assign led_3bits_tri_o[0] = heartbeat_count[25];
  assign led_3bits_tri_o[1] = pass_latched;
  assign led_3bits_tri_o[2] = fail_latched;
endmodule
