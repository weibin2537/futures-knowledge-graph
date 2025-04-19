import os
import json
import logging
import yaml
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Config:
    """期货业务知识库配置管理"""
    
    DEFAULT_CONFIG = {
        # 数据库配置
        "database": {
            "neo4j_uri": "bolt://localhost:7687",
            "neo4j_user": "neo4j",
            "neo4j_password": "password"
        },
        
        # 模型配置
        "models": {
            "entity_model_path": None,  # 实体识别模型路径，如果为None则使用规则匹配
            "relation_model_path": None  # 关系分类模型路径，如果为None则使用规则匹配
        },
        
        # 数据处理配置
        "processing": {
            "batch_size": 5,  # 批处理大小
            "max_text_length": 10000,  # 最大处理文本长度
            "supported_extensions": [".pdf", ".docx", ".doc", ".xlsx", ".xls", ".md", ".markdown"]
        },
        
        # 系统路径配置
        "paths": {
            "upload_dir": "./uploads",  # 上传文件目录
            "output_dir": "./output",  # 输出目录
            "model_dir": "./models",  # 模型目录
            "temp_dir": "./temp"  # 临时文件目录
        },
        
        # API配置
        "api": {
            "host": "0.0.0.0",
            "port": 8000,
            "debug": False,
            "reload": False,
            "cors_origins": ["*"],
            "token_expire_minutes": 60 * 24  # 1天
        },
        
        # 日志配置
        "logging": {
            "level": "INFO",
            "file": "futures_knowledge.log",
            "max_size": 10 * 1024 * 1024,  # 10MB
            "backup_count": 5
        },
        
        # 实体和关系类型配置
        "entity_types": [
            {"name": "CONTRACT", "label": "合约", "color": "#2980b9"},
            {"name": "RULE", "label": "规则", "color": "#c0392b"},
            {"name": "DATE", "label": "日期", "color": "#27ae60"},
            {"name": "MARGIN", "label": "保证金比例", "color": "#f39c12"},
            {"name": "POSITION_LIMIT", "label": "持仓限额", "color": "#8e44ad"},
            {"name": "VARIETY", "label": "品种", "color": "#16a085"},
            {"name": "TERM", "label": "术语", "color": "#7f8c8d"}
        ],
        
        "relation_types": [
            {"name": "APPLY_TO", "label": "适用于", "description": "规则适用于合约"},
            {"name": "DEFINED_IN", "label": "定义于", "description": "在文档中定义"},
            {"name": "MENTIONED_IN", "label": "提及于", "description": "在文档中提及"},
            {"name": "EXPLAINED_IN", "label": "解释于", "description": "在文档中解释"},
            {"name": "HAS_MARGIN", "label": "保证金比例", "description": "具有保证金比例"},
            {"name": "HAS_LIMIT", "label": "持仓限额", "description": "具有持仓限额"},
            {"name": "BELONGS_TO", "label": "属于", "description": "属于品种"},
            {"name": "EFFECTIVE_FROM", "label": "生效于", "description": "从日期生效"},
            {"name": "DELIVERY_ON", "label": "交割于", "description": "在日期交割"}
        ]
    }
    
    def __init__(self, config_path=None, env=None):
        """
        初始化配置
        
        Args:
            config_path: 配置文件路径，可选
            env: 环境名称（如开发、测试、生产），可选
        """
        self.config = self.DEFAULT_CONFIG.copy()
        
        # 如果指定了配置文件，加载配置
        if config_path and os.path.exists(config_path):
            self._load_config(config_path, env)
        
        # 创建必要的目录
        self._create_directories()
        
        logger.info("配置加载完成")
    
    def _load_config(self, config_path, env=None):
        """
        从文件加载配置
        
        Args:
            config_path: 配置文件路径
            env: 环境名称，可选
        """
        try:
            if config_path.endswith(".json"):
                with open(config_path, "r", encoding="utf-8") as f:
                    file_config = json.load(f)
            elif config_path.endswith((".yaml", ".yml")):
                with open(config_path, "r", encoding="utf-8") as f:
                    file_config = yaml.safe_load(f)
            else:
                logger.error(f"不支持的配置文件格式: {config_path}")
                return
            
            # 如果指定了环境，加载特定环境的配置
            if env and env in file_config:
                env_config = file_config[env]
                self._update_config(env_config)
                logger.info(f"已加载{env}环境配置")
            else:
                # 否则加载全局配置
                self._update_config(file_config)
                logger.info("已加载全局配置")
        
        except Exception as e:
            logger.error(f"加载配置文件失败: {str(e)}")
    
    def _update_config(self, new_config):
        """
        递归更新配置
        
        Args:
            new_config: 新配置
        """
        for key, value in new_config.items():
            if key in self.config and isinstance(self.config[key], dict) and isinstance(value, dict):
                # 递归更新嵌套字典
                self._update_config_dict(self.config[key], value)
            else:
                # 直接更新值
                self.config[key] = value
    
    def _update_config_dict(self, target_dict, source_dict):
        """
        递归更新字典
        
        Args:
            target_dict: 目标字典
            source_dict: 源字典
        """
        for key, value in source_dict.items():
            if key in target_dict and isinstance(target_dict[key], dict) and isinstance(value, dict):
                # 递归更新嵌套字典
                self._update_config_dict(target_dict[key], value)
            else:
                # 直接更新值
                target_dict[key] = value
    
    def _create_directories(self):
        """创建必要的目录"""
        directories = [
            self.config["paths"]["upload_dir"],
            self.config["paths"]["output_dir"],
            self.config["paths"]["model_dir"],
            self.config["paths"]["temp_dir"]
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"创建目录: {directory}")
    
    def get(self, key, default=None):
        """
        获取配置值
        
        Args:
            key: 配置键，支持点号分隔的路径
            default: 默认值
            
        Returns:
            配置值
        """
        try:
            # 处理点号分隔的路径
            if "." in key:
                parts = key.split(".")
                value = self.config
                for part in parts:
                    value = value[part]
                return value
            else:
                return self.config[key]
        except KeyError:
            return default
    
    def set(self, key, value):
        """
        设置配置值
        
        Args:
            key: 配置键，支持点号分隔的路径
            value: 配置值
        """
        try:
            # 处理点号分隔的路径
            if "." in key:
                parts = key.split(".")
                config = self.config
                for part in parts[:-1]:
                    if part not in config:
                        config[part] = {}
                    config = config[part]
                config[parts[-1]] = value
            else:
                self.config[key] = value
            
            logger.debug(f"设置配置: {key} = {value}")
        except Exception as e:
            logger.error(f"设置配置失败: {key} - {str(e)}")
    
    def save(self, config_path):
        """
        保存配置到文件
        
        Args:
            config_path: 配置文件路径
        """
        try:
            directory = os.path.dirname(config_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            
            if config_path.endswith(".json"):
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(self.config, f, ensure_ascii=False, indent=2)
            elif config_path.endswith((".yaml", ".yml")):
                with open(config_path, "w", encoding="utf-8") as f:
                    yaml.dump(self.config, f, allow_unicode=True, default_flow_style=False)
            else:
                logger.error(f"不支持的配置文件格式: {config_path}")
                return False
            
            logger.info(f"配置已保存到: {config_path}")
            return True
        
        except Exception as e:
            logger.error(f"保存配置失败: {str(e)}")
            return False
    
    def get_entity_types(self):
        """获取实体类型配置"""
        return self.config.get("entity_types", [])
    
    def get_relation_types(self):
        """获取关系类型配置"""
        return self.config.get("relation_types", [])
    
    def get_database_config(self):
        """获取数据库配置"""
        return self.config.get("database", {})
    
    def get_api_config(self):
        """获取API配置"""
        return self.config.get("api", {})
    
    def get_models_config(self):
        """获取模型配置"""
        return self.config.get("models", {})
    
    def get_processing_config(self):
        """获取数据处理配置"""
        return self.config.get("processing", {})
    
    def get_paths_config(self):
        """获取路径配置"""
        return self.config.get("paths", {})
    
    def get_logging_config(self):
        """获取日志配置"""
        return self.config.get("logging", {})


# 创建全局配置实例
def create_config(config_path=None, env=None):
    """
    创建配置实例
    
    Args:
        config_path: 配置文件路径，可选
        env: 环境名称，可选
        
    Returns:
        Config: 配置实例
    """
    return Config(config_path, env)


# 示例使用
if __name__ == "__main__":
    # 创建默认配置
    config = create_config()
    
    # 打印部分配置
    print("数据库配置:")
    print(f"  URI: {config.get('database.neo4j_uri')}")
    print(f"  用户: {config.get('database.neo4j_user')}")
    
    print("\n实体类型:")
    for entity_type in config.get_entity_types():
        print(f"  {entity_type['name']}: {entity_type['label']} ({entity_type['color']})")
    
    print("\n关系类型:")
    for relation_type in config.get_relation_types():
        print(f"  {relation_type['name']}: {relation_type['label']} - {relation_type['description']}")
    
    # 修改配置
    config.set("api.port", 8080)
    config.set("logging.level", "DEBUG")
    
    # 保存配置
    config.save("./config.yaml")