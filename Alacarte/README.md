# 复现：Optimal Structured Light a la Carte
[原文链接](https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1697.pdf)  
[中文博客](https://namisntimpot.github.io/posts/research/strutured-light/2-optimal-structured-light--a-la-carte/)  

|硬件|使用blender模拟，未加参数。相机和投影仪根据blender的设定计算理想内外参。分辨率均$800\times 600$|  
|:-:|:-:|  
|代码|[这里](tmp)|  
|备注|未考虑几何范围约束，未考虑投影仪defocus。虽然有所实现，但不完全或没测试。$\mu$=300，迭代1000轮。提前采样250组条件\<T, A ,E>用于evaluation|  
  
|场景|groundtruth depth|  
|:-:|:-:|  
|![regular scene](https://lijiaheng-picture-host.oss-cn-beijing.aliyuncs.com/notebook-images/202408111200960.png)|![真实深度](https://lijiaheng-picture-host.oss-cn-beijing.aliyuncs.com/notebook-images/gt_depth.png)|  

## 结果
  
|条件|code matrix|相似度矩阵|深度图|  
|:-:|:-:|:-:|:-:|  
|patterns 4<br>maxfreq 4<br>$\epsilon$ 0|![4-4-0-code](https://lijiaheng-picture-host.oss-cn-beijing.aliyuncs.com/notebook-images/code_matrix_vis.png)|![4-4-0-sim](https://lijiaheng-picture-host.oss-cn-beijing.aliyuncs.com/notebook-images/sim.png)|![4-4-0-depth](https://lijiaheng-picture-host.oss-cn-beijing.aliyuncs.com/notebook-images/depth_lightoff.png)|  
|patterns 4<br>maxfreq 4<br>$\epsilon$ 3|![4-4-3-code](https://lijiaheng-picture-host.oss-cn-beijing.aliyuncs.com/notebook-images/202408110038692.png)|![4-4-3-sim](https://lijiaheng-picture-host.oss-cn-beijing.aliyuncs.com/notebook-images/202408110039357.png)|![4-4-3-depth](https://lijiaheng-picture-host.oss-cn-beijing.aliyuncs.com/notebook-images/202408110039973.png)|  
|patterns 4<br>maxfreq 8<br>$\epsilon$ 0|![4-8-0-code](https://lijiaheng-picture-host.oss-cn-beijing.aliyuncs.com/notebook-images/202408110040568.png)|![4-8-0-sim](https://lijiaheng-picture-host.oss-cn-beijing.aliyuncs.com/notebook-images/202408110040769.png)|![4-8-0-depth](https://lijiaheng-picture-host.oss-cn-beijing.aliyuncs.com/notebook-images/202408110041619.png)|  
|patterns 4<br>maxfreq 8<br>$\epsilon$ 3|![4-8-3-code](https://lijiaheng-picture-host.oss-cn-beijing.aliyuncs.com/notebook-images/202408110042826.png)|![4-8-3-sim](https://lijiaheng-picture-host.oss-cn-beijing.aliyuncs.com/notebook-images/202408110042463.png)|![4-8-3-depth](https://lijiaheng-picture-host.oss-cn-beijing.aliyuncs.com/notebook-images/202408110043166.png)|  
|patterns 4<br>maxfreq 16<br>$\epsilon$ 0|![4-16-0-code](https://lijiaheng-picture-host.oss-cn-beijing.aliyuncs.com/notebook-images/202408110107306.png)|![4-16-0-sim](https://lijiaheng-picture-host.oss-cn-beijing.aliyuncs.com/notebook-images/202408110107649.png)|![4-16-0-depth](https://lijiaheng-picture-host.oss-cn-beijing.aliyuncs.com/notebook-images/202408110108955.png)|  
|patterns 4<br>maxfreq 16<br>$\epsilon$ 3|![4-16-3-code](https://lijiaheng-picture-host.oss-cn-beijing.aliyuncs.com/notebook-images/202408110109866.png)|![4-16-3-sim](https://lijiaheng-picture-host.oss-cn-beijing.aliyuncs.com/notebook-images/202408110109819.png)|![4-16-3-depth](https://lijiaheng-picture-host.oss-cn-beijing.aliyuncs.com/notebook-images/202408110110894.png)|  
|patterns 4<br>maxfreq $\infty$<br>$\epsilon$ 0|![4-inf-0-code](https://lijiaheng-picture-host.oss-cn-beijing.aliyuncs.com/notebook-images/202408110111047.png)|![4-inf-0-sim](https://lijiaheng-picture-host.oss-cn-beijing.aliyuncs.com/notebook-images/202408110112153.png)|![4-inf-0-depth](https://lijiaheng-picture-host.oss-cn-beijing.aliyuncs.com/notebook-images/202408110112534.png)|  
|patterns 5<br>maxfreq 4<br>$\epsilon$ 0|![5-4-0-code](https://lijiaheng-picture-host.oss-cn-beijing.aliyuncs.com/notebook-images/202408110113542.png)|![5-4-0-sim](https://lijiaheng-picture-host.oss-cn-beijing.aliyuncs.com/notebook-images/202408110114627.png)|![5-4-0-depth](https://lijiaheng-picture-host.oss-cn-beijing.aliyuncs.com/notebook-images/202408110114091.png)|  
|sinshift<br>freq=4<br>shifts=4|![sin-code](https://lijiaheng-picture-host.oss-cn-beijing.aliyuncs.com/notebook-images/202408110116679.png)|![sin-sim](https://lijiaheng-picture-host.oss-cn-beijing.aliyuncs.com/notebook-images/202408110117119.png)|![sin-depth](https://lijiaheng-picture-host.oss-cn-beijing.aliyuncs.com/notebook-images/202408110117878.png)|  

## 问题
关于损失函数值的优化，可能是我的实现或者参数设置还有点问题，虽然能够优化、数值能够下降、得到的pattern效果看起来也不错，但损失函数值挺大的，下降幅度也很小，尤其是最高频率较低、容忍度$\epsilon$小的情况下。  
  
感觉其实也比较合理，correct是 $\frac{\exp(\mu ZNCC(o_q, matched(o_q)))}{\exp(\sum_{r=1}^NZNCC(o_q, c_r))}$，所以越多和matched code相似的越多这个correct就越小。限制了最大频率后，相近的几列大概率会比较相似，而哪怕只有10个比较相似的code，correct都只有1/10了，每列都有这样比较相似的，总的算下来error就大了。但error数值大，最后优化出来的Pattern效果应该还是可以的。  
  
提高最大频率，提高容忍度$\epsilon$，就能降低error数值。这倒是符合预期。但没看见论文中那种快速下降并收敛的曲线。不过不管怎样，1000轮（一般500轮以内就行）迭代以内，优化的pattern就基本稳定了。  
  
也可能是我的实现还是有点小问题，或者参数设置不对，比如我给的随机环境光、噪声的影响都挺小。  