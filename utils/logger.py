"""
日志配置模块
提供统一的日志管理，同时输出到控制台和文件
"""

import logging
from dify_plugin.config.logger_format import plugin_logger_handler


def get_logger(name: str = "zep") -> logging.Logger:
    """
    获取日志器（如果不存在则创建）

    Args:
        name: 日志器名称

    Returns:
        日志器实例
    """
    # 使用自定义处理器设置日志
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(plugin_logger_handler)
    return logger
