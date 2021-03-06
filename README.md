# guarder
一种面向运维的PLUG IN式异常探测框架
## 核心功能
在不降低异常探测准确度的前提下，有效减少误报
## 背景
异常探测在运维领域中十分重要。然而在实际运维活动中，面对实时的多维数据，直接使用或者简单改造现有的异常探测算法，往往会有大量的误报点，这会频繁的打扰运维人员，影响其工作效率，甚至让运维人员放弃使用异常探测系统。注意到运维领域的一个数据事实：许多异常往往是连续异常点构成的异常时间段，我们提出了一个通用的机器学习框架，通过结合不同异常探测算法的优点，并利用时间连续性原理，能够学习到异常模式的“开端”并及时报警，同时提出的框架能够大幅度的减小误报；<br> 
  1. 我们提出的框架具有很好的灵活性和扩展性，能够针对不同的实际问题，嵌入不同的异常探测算法，从而构建出针对实际问题的专门的异常探测系统。<br>
  2. 提出的框架能够有效的权衡误报率和异常探测精度，通过UCI的实际数据和benchmark，以及实际的工业数据应用，可以证实提出的框架具有良好的表现，十分适合实际的运维领域的异常探测问题。<br>
## 框架的优势
1. 强调了运维领域中，在具有异常时间段的多维数据流进行异常探测的问题。<br>
2. 强调和介绍了将监督异常探测算法和无监督探测算法结合的方案。<br>
3. 提出了一种新颖的灵活的异常探测框架，能够将已有的异常探测算法进行集成，使之更适合实际应用。<br>
4. 提出的框架能够有效的减少误报率，实现误报率和异常探测精度的权衡；通过有效识别异常时间段的开端，及时报警并通知运维人员，实现智能运维。<br>
## 核心思想
以往的算法都是针对单个的数据点进行异常探测，导致异常探测结果不够稳定，容易产生波动和误报。我们的算法引入了滑动窗口（窗口大小较小），通过机器学习算法学习多个异常探测器对于窗口中连续几个数据点的异常打分的规律，从而有效减少误报和波动。
###
Springer Link: https://link.springer.com/chapter/10.1007/978-981-13-2206-8_25
