"""WorldForge-AI 算法内核包。

做什么
    承载 10 个零第三方依赖的数学内核模块，按 docs/ALGORITHM_KERNEL.md 实现。

不变量
    本包不引入任何 import 副作用：子模块之间只存在 numeric -> linalg -> 上层
    的单向依赖，导入任意一个子模块都不会触发循环导入。

约定
    纯标准库（math / random / cmath / itertools / fractions 等），无 numpy。
    矩阵统一为 list[list[float]]，行主序，M[i][j] 为第 i 行第 j 列。
"""
