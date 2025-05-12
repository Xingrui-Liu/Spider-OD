# Spider-OD
异常点检测算法-基于蜘蛛捕食飞航行为建立
# Spider-OD

**Spider-OD** 是一款轻量级的仿生异常检测库，灵感来自群居蜘蛛的 **“织网 → 振动 → 捕食 → 飞航”** 协同策略。  

---

## 项目简介
- **动态邻域**：初始化时仅需给出一个近邻阈值 `Dthr`，模型会在振动传播过程中自适应调节 `k`。  
- **能量驱动**：样本能量随振动传播不断衰减 / 增长；当能量高于或低于阈值即触发捕食或飞航动作，实现异常点捕获与模型自进化。  
- **时间复杂度**：`O(n log n + n k_max)`，内存占用约 `O(n)`。
- **接口友好**：提供 `fit-predict-decision_function`，可与 scikit-learn / PyOD 等框架无缝联动。

- **文件详情**
- **一、主程序与核心算法**
-	main.py：命令行入口，完成数据加载、参数解析与实验流程调度
-	SpiderODv5.py：Spider-OD 算法实现
- **二、辅助工具**
-	dataPreprocessing.py：数据清洗、标准化、随机划分等通用预处理函数


---

## 技术栈说明
| 层次 | 依赖 / 工具 |
|------|-------------|
| 核心数值计算 | **NumPy** • SciPy |
| 数据集 & 指标 | **scikit-learn** |
| 基线算法对比 | **PyOD** |
| 打包 / CI | Poetry • pre-commit • GitHub Actions |

---

## License
本项目以 **MIT License** 开源，详见仓库中的 [`LICENSE`](LICENSE) 文件。
