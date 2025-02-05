该仓库由两个分支组成，main分支的主要内容为课程设计的代码及报告，mid分支为课程作业的代码及报告。

- mid 分支
  - 主要进行了LeNet-5的模型优化方法的比较分析
  - 对LeNet-5模型优化方法进行比较分析或改进；
  - 比较三种的优化方法（动量SGD, AdaDelta, Adam等）；
  - 在MNIST/Cifar-10数据集分析比较不同优化方法的收敛行为和泛化性。

- main分支
  - 代码源于[kentaroy47/](https://github.com/kentaroy47/vision-transformers-cifar10)
  - 基于源代码，改进了Swin模型（文件名swin-revisedd.py）
  - 深入分析模型的性能差异：通过在 CIFAR-10 数据集上对 ViT small 和 Swin Transformer 进行实验，系统分析其在图像分类任务中的表现，探索局部注意力机制、分层设计等关键技术对模型性能的影响。 
  - 在 Swin Transformer 的基础上，引入混合注意力机制（局部卷积 + 全局注意力）以及轻量化设计（稀疏连接和深度可分离卷积），以提高模型的计算效率和小数据集上的适应性。 
  - 通过消融实验、对比实验分析，验证改进模型在参数量、计算复杂度和分类准确率等方面的优势。 

