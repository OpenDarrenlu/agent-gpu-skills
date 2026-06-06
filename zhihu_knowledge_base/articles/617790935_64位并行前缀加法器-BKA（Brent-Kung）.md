# 64位并行前缀加法器-BKA（Brent-Kung）

**作者**: SnoopyBug划水划呀划~

**原文链接**: https://zhuanlan.zhihu.com/p/617790935

---

BKA64树形示意图

对于BK型并行前缀加法器，关键是树形结构的构造。上图中每个黑点表示做一次运算: 
(
𝑔
1
,
𝑝
1
)
∘
(
𝑔
2
,
𝑝
2
)
=
(
𝑔
1
+
𝑝
1
⋅
𝑔
2
,
𝑝
1
⋅
𝑝
2
)

首先通过计算p和g，再按照树形结构计算G和P，根据G、P、cin计算每一位的进位信号C，从而得到s与cout。

组合逻辑较复杂，部分代码能够循环生成，剩余的直接手动生成。源码过长，详见仓库：

// ! 注意g1和g2的位置与引用模块时的输入对应，弄错了就g了，呜呜~
module black (input g1, p1, g2, p2, output gout, pout);
    assign gout = g2 | p2 & g1;
    assign pout = p2 & p1;
endmodule


module BKA64 (
    input [63:0] a, b,
    input cin,
    output [63:0] s,
    output cout
    );

genvar i;

// T=0
wire [63:0] g, p;
assign g = a & b;
assign p = a ^ b;

// T=1
wire [31:0] g_1, p_1;
generate
    for (i=0; i<32; i=i+1) begin
        black black1(
            .g1(g[i*2]),
            .p1(p[i*2]),
            .g2(g[i*2+1]),
            .p2(p[i*2+1]),
            .gout(g_1[i]),
            .pout(p_1[i])
        );
    end
endgenerate

// T=2
wire [15:0] g_2, p_2;
generate
    for (i=0; i<16; i=i+1) begin
        black black2(
            .g1(g_1[i*2]),
            .p1(p_1[i*2]),
            .g2(g_1[i*2+1]),
            .p2(p_1[i*2+1]),
            .gout(g_2[i]),
            .pout(p_2[i])
        );
    end
endgenerate

// T=3
......

// T=4
......

// T=5
......

// T=6
wire g_6, p_6;
black black6(
    .g1(g_5[0]),
    .p1(p_5[0]),
    .g2(g_5[1]),
    .p2(p_5[1]),
    .gout(g_6),
    .pout(p_6)
);

// T=7
wire g_7, p_7;
black black7(
    .g1(g_5[0]),
    .p1(p_5[0]),
    .g2(g_4[2]),
    .p2(p_4[2]),
    .gout(g_7),
    .pout(p_7)
);

// T=8
wire [2:0] g_8, p_8;
black black8_1(
    .g1(g_4[0]),
    .p1(p_4[0]),
    .g2(g_3[2]),
    .p2(p_3[2]),
    .gout(g_8[0]),
    .pout(p_8[0])
);
black black8_2(
    .g1(g_5[0]),
    .p1(p_5[0]),
    .g2(g_3[4]),
    .p2(p_3[4]),
    .gout(g_8[1]),
    .pout(p_8[1])
);
black black8_3(
    .g1(g_7),
    .p1(p_7),
    .g2(g_3[6]),
    .p2(p_3[6]),
    .gout(g_8[2]),
    .pout(p_8[2])
);

// T=9
......

// T=10
......

// T=11
......

// P, G
wire [63:0] P, G;
generate
    for (i=0; i<31; i=i+1) begin
        assign G[2*i+2] = g_11[i];
        assign P[2*i+2] = p_11[i];
    end
    for (i=0; i<15; i=i+1) begin
        assign G[4*i+5] = g_10[i];
        assign P[4*i+5] = p_10[i];
    end
    for (i=0; i<7; i=i+1) begin
        assign G[8*i+11] = g_9[i];
        assign P[8*i+11] = p_9[i];
    end
    for (i=0; i<3; i=i+1) begin
        assign G[16*i+23] = g_8[i];
        assign P[16*i+23] = p_8[i];
    end
    for (i=0; i<1; i=i+1) begin
        assign G[32*i+47] = g_7;
        assign P[32*i+47] = p_7;
    end
endgenerate
assign G[0] = g[0];
assign P[0] = p[0];
assign G[1] = g_1[0];
assign P[1] = p_1[0];
assign G[3] = g_2[0];
assign P[3] = p_2[0];
assign G[7] = g_3[0];
assign P[7] = p_3[0];
assign G[15] = g_4[0];
assign P[15] = p_4[0];
assign G[31] = g_5[0];
assign P[31] = p_5[0];
assign G[63] = g_6;
assign P[63] = p_6;

// C, s
wire [64:0] C;
generate
    assign C[0] = cin;
    for (i=0; i<64; i=i+1) begin
        assign C[i+1] = G[i] | P[i]&C[0];
    end
endgenerate
assign cout = C[64];
assign s = C[63:0] ^ p;

endmodule

