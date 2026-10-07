"""WorldForge 的统一异常层级。

作者：晨星
"""


class WorldforgeError(Exception):
    """所有可预期领域异常的基类。"""


class ConfigError(WorldforgeError):
    """配置缺失、冲突或取值非法。"""


class BackendUnavailable(WorldforgeError):
    """生产后端无法导入或连接。"""


class DimensionMismatch(WorldforgeError):
    """向量维度或索引指纹不匹配。"""


class VectorStoreError(WorldforgeError):
    """向量索引损坏或后端操作失败。"""


class DocNotFound(WorldforgeError):
    """请求的文档不存在。"""


class IngestError(WorldforgeError):
    """文档摄入失败。"""


class ToolError(WorldforgeError):
    """工具调用的统一异常出口。"""


class SafeEvalError(ToolError):
    """安全算术求值失败。"""


class LLMError(WorldforgeError):
    """语言模型后端调用失败。"""


class PlanError(WorldforgeError):
    """计划不合法或无法执行。"""


class EvalGateError(WorldforgeError):
    """评估结果未达到发布门限。"""


__all__ = [
    "WorldforgeError",
    "ConfigError",
    "BackendUnavailable",
    "DimensionMismatch",
    "VectorStoreError",
    "DocNotFound",
    "IngestError",
    "ToolError",
    "SafeEvalError",
    "LLMError",
    "PlanError",
    "EvalGateError",
]
__author__ = "晨星"
