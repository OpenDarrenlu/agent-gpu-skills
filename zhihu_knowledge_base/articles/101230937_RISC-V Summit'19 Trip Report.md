# RISC-V Summit'19 Trip Report

**作者**: 涂锋斌​清华大学 微电子与纳电子学系博士

**原文链接**: https://zhuanlan.zhihu.com/p/101230937

---

12月份的时候，谢老师安排我去一趟湾区参加RISC-V峰会。这不是一个学术会议，更像是一个行业交流会。所有和这个生态相关的从业人员齐聚一趟，讨论到目前为止大家在RISC-V上的进展程度。从前在国内的时候，大多只能从网上通过文字图片来了解这个圈子，猜测它可能是怎样的，来到这里后有幸能够亲眼看看RISC-V这个圈子，形成一种直观立体的认识。本文内容节选自我在SEAL提交的trip report。

RISC-V Summit'19 Trip Report, by Fengbin Tu

San Jose, CA, December 10-12, 2019




Outline

In this report, I will not go into too many details of each talk. I prefer talking about my understanding on the RISC-V ecosystem. The report is organized as follows:

The current status of RISC-V ecosystem.
RISC-V's development and roadmap.
Why we need RISC-V?
Western Digital and OmniXtend.
SiFive and its business model.




In conclusion, I think the most important two keywords for RISC-V are customization and low cost.




The current status of RISC-V ecosystem.




Calista Redmond, CEO at RISC-V Foundation, presented a welcome talk on Exponential Progress across Industries and Around the World with RISC-V. Calista discussed the progress spanning the global RISC-V community in collaboration, innovation, and adoption of RISC-V across numerous industries and around the world.










RISC-V's development and roadmap.




Krste Asanovic (Professor | Chief Architect at UC Berkeley | SiFive) gave a keynote titled “State of the Union”.




RISC-V came from the Berkelye Par Lab. They needed a simple, efficient and extensible ISA in 2010.










The following is the RISC-V Timeline.










What’s different about RISC-V? The following slides summarize its advantage from the ISA aspect.










Why is RISC-V so popular? This is because of the new business model.

ISA -> Vendor/Build own core -> Add extension freely.










Why we need RISC-V?




At the summit, most of the talks were given by companies. RISC-V is cheapness and extensibility help them design their own product with both lower cost. Lowering the cost is the business principle for all companies. RISC-V gives them a good alternative to ARM or Intel. Meanwhile, with so many members, RISC-V itself also grows rapidly.




Here I share three talks about extending RISC-V for security, signal processing and AI applications.




In the talk “Architectural Extensions for a RISC-V Processor for Embedded Security”, Tariq Kurd (CPU Architect, Huawei UK) described how they started from a standard RISC-V processor and developed a world-class embedded processor suitable for running security applications, such as iSim for embedded SIM cards.










Zdenek Prikryl, CTO at Codasip, presented an instruction-set extension to the open-source RISC-V ISA (RV32IM) dedicated to ultra-low power (ULP) software-defined wireless IoT transceivers. The custom instructions are tailored to the needs of 8/16/32-bit integer complex arithmetic typically required by quadrature modulations. The proposed extension occupies only 3 major opcodes and most instructions are designed to come at a near-zero hardware and energy cost. Using Codasip Studio, an instruction accurate (IA) model of the new architecture is used to evaluate four IoT baseband processing test benches: FSK demodulation, LoRa preamble detection, 32-bit FFT and CORDIC algorithm. Results show an average energy efficiency improvement of more than 35% with up to 50% obtained for the LoRa preamble detection algorithm.










Karthik Wali, Staff Digital Design Engineer at LG Electronics, introduced a “Scalable, Configurable Neural Network Accelerator Based on RISC-V Core”. As I’m familiar with this field, I know that they use RISC-V cores as the main controller for their CNN processing SoC. RISC-V reduces their cost for designing/buying cores and verification, and helps them quickly build a stable system. The following slides summarizes why they use RISC-V.













I feel a little surprised that Google also involves in the RISC-V, and is even building an end-to-end open source solution for RISC-V processor verification. This talk makes me believe that Google is also doing some great things in machine learning for EDA, also mentioned in Jeff Dean’s recent arXiv paper. I guess hardware design is being paid great attention in Google nowadays. They might be co-designing algorithm and hardware to build AI systems in this post-Moore’s law era.



















Western Digital and OmniXtend.




WD might be the most active company in this year’s summit. They presented OmniXtend that utilizes RISC-V, to enable innovation in next-generation data-centric computing.

In such a heterogeneous system (see Figure below), machine-learning accelerators, custom ASICs, FPGAs, CPUs and GPUs should all reside in the same memory domain and share memory in a coherent way. With the help of RISC-V, ethernet protocol is replaced with OmniXtend protocol to handle memory coherency traffic.




























SiFive and its business model.




SiFive is another star company at the summit. It was founded in 2015 by Krste Asanovic, Yunsup Lee, and Andrew Waterman, three researchers from the University of California Berkeley. SiFive is the first fabless semiconductor company to build customized silicon based on the free and open RISC-V instruction set architecture.



















SiFive's business model is based on designing custom computer chips for other businesses, as shown in the following figures (screenshot from their website). It's really cool. I just concern that the SoC's quality is largely determined by the open-sourced IP cores, and thus verification is quite important in the design flow.









