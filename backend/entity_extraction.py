import re
import logging
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EntityExtractor:
    """实体提取器，用于从文本中识别业务领域实体"""
    
    def __init__(self, model_path=None):
        """
        初始化实体提取器
        
        Args:
            model_path: 预训练模型路径，如果为None则使用规则匹配
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.ner_pipeline = None
        
        # 实体类型映射
        self.entity_types = {
            "CONTRACT": "合约",
            "RULE": "规则",
            "DATE": "日期",
            "MARGIN": "保证金比例",
            "POSITION_LIMIT": "持仓限额",
            "VARIETY": "品种",
            "TERM": "术语"
        }
        
        # 初始化模型或规则匹配器
        self._initialize()
    
    def _initialize(self):
        """初始化实体提取模型或规则匹配器"""
        if self.model_path and os.path.exists(self.model_path):
            try:
                # 加载预训练模型
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)
                self.ner_pipeline = pipeline(
                    "ner",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if torch.cuda.is_available() else -1
                )
                logger.info(f"已加载NER模型: {self.model_path}")
            except Exception as e:
                logger.error(f"加载NER模型失败: {str(e)}")
                logger.info("将使用规则匹配进行实体提取")
                self.model_path = None
        else:
            logger.info("未指定模型路径或模型不存在，将使用规则匹配进行实体提取")
    
    def extract_entities(self, text):
        """
        从文本中提取实体
        
        Args:
            text: 待提取实体的文本
            
        Returns:
            list: 提取的实体列表，每个实体为字典格式
        """
        if self.ner_pipeline:
            return self._extract_with_model(text)
        else:
            return self._extract_with_rules(text)
    
    def _extract_with_model(self, text):
        """使用预训练模型提取实体"""
        try:
            # 对长文本进行切分，避免超出模型输入长度限制
            max_length = self.tokenizer.model_max_length - 2  # 预留CLS和SEP
            chunks = self._split_text(text, max_length)
            
            all_entities = []
            for chunk in chunks:
                results = self.ner_pipeline(chunk)
                
                # 处理结果，合并相同类型的相邻词元
                entities = []
                current_entity = None
                
                for result in results:
                    # 如果是B-开头的标签，表示一个新实体的开始
                    if result["entity"].startswith("B-"):
                        # 如果当前有正在处理的实体，先保存
                        if current_entity:
                            entities.append(current_entity)
                        
                        # 创建新实体
                        entity_type = result["entity"][2:]  # 去除B-前缀
                        current_entity = {
                            "type": entity_type,
                            "label": self.entity_types.get(entity_type, entity_type),
                            "text": result["word"],
                            "start": result["start"],
                            "end": result["end"],
                            "score": result["score"]
                        }
                    # 如果是I-开头的标签，表示当前实体的继续
                    elif result["entity"].startswith("I-") and current_entity:
                        entity_type = result["entity"][2:]
                        if entity_type == current_entity["type"]:
                            current_entity["text"] += result["word"]
                            current_entity["end"] = result["end"]
                            # 更新置信度为平均值
                            current_entity["score"] = (current_entity["score"] + result["score"]) / 2
                
                # 保存最后一个实体
                if current_entity:
                    entities.append(current_entity)
                
                all_entities.extend(entities)
            
            return all_entities
        except Exception as e:
            logger.error(f"模型提取实体出错: {str(e)}")
            # 出错时回退到规则匹配
            return self._extract_with_rules(text)
    
    def _extract_with_rules(self, text):
        """使用规则匹配提取实体"""
        entities = []
        
        # 合约代码规则匹配（如IF2406）
        contract_pattern = r'\b([A-Z]{1,2}\d{4})\b'
        for match in re.finditer(contract_pattern, text):
            entities.append({
                "type": "CONTRACT",
                "label": self.entity_types["CONTRACT"],
                "text": match.group(0),
                "start": match.start(),
                "end": match.end(),
                "score": 1.0  # 规则匹配设置固定置信度
            })
        
        # 日期匹配（如2024年6月15日、2024-06-15）
        date_patterns = [
            r'(\d{4}年\d{1,2}月\d{1,2}日)',
            r'(\d{4}-\d{1,2}-\d{1,2})',
            r'(\d{4}/\d{1,2}/\d{1,2})'
        ]
        
        for pattern in date_patterns:
            for match in re.finditer(pattern, text):
                entities.append({
                    "type": "DATE",
                    "label": self.entity_types["DATE"],
                    "text": match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                    "score": 1.0
                })
        
        # 保证金比例匹配（如8%、0.08）
        margin_patterns = [
            r'保证金比例[为是:]?\s*(\d+(?:\.\d+)?%)',
            r'保证金比例[为是:]?\s*(\d+(?:\.\d+)?)'
        ]
        
        for pattern in margin_patterns:
            for match in re.finditer(pattern, text):
                entities.append({
                    "type": "MARGIN",
                    "label": self.entity_types["MARGIN"],
                    "text": match.group(1),
                    "start": match.start(1),
                    "end": match.end(1),
                    "score": 1.0
                })
        
        # 持仓限额匹配
        position_pattern = r'持仓限额[为是:]?\s*(\d+)(?:手|张|单位)?'
        for match in re.finditer(position_pattern, text):
            entities.append({
                "type": "POSITION_LIMIT",
                "label": self.entity_types["POSITION_LIMIT"],
                "text": match.group(1),
                "start": match.start(1),
                "end": match.end(1),
                "score": 1.0
            })
        
        # 品种匹配
        variety_list = ["沪铜", "豆粕", "白糖", "螺纹钢", "沪深300", "中证500", "原油", "黄金", "铁矿石"]
        for variety in variety_list:
            for match in re.finditer(variety, text):
                entities.append({
                    "type": "VARIETY",
                    "label": self.entity_types["VARIETY"],
                    "text": match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                    "score": 1.0
                })
        
        # 规则编号匹配（如RULE_2024_001）
        rule_pattern = r'\b(RULE_\d{4}_\d{3})\b'
        for match in re.finditer(rule_pattern, text):
            entities.append({
                "type": "RULE",
                "label": self.entity_types["RULE"],
                "text": match.group(0),
                "start": match.start(),
                "end": match.end(),
                "score": 1.0
            })
        
        return entities
    
    def _split_text(self, text, max_length):
        """
        将长文本切分为适合模型处理的长度
        
        Args:
            text: 待切分的文本
            max_length: 每个切片的最大长度
            
        Returns:
            list: 文本切片列表
        """
        # 使用句号和换行符切分
        sentences = re.split(r'(?<=[。.!?！？\n])', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # 如果句子本身超长，按最大长度切分
            if len(sentence) > max_length:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                # 按max_length直接切分句子
                for i in range(0, len(sentence), max_length):
                    chunks.append(sentence[i:i+max_length])
            # 如果加上这个句子会超长，先保存当前块，再开始新块
            elif len(current_chunk) + len(sentence) > max_length:
                chunks.append(current_chunk)
                current_chunk = sentence
            # 否则，将句子添加到当前块
            else:
                current_chunk += sentence
        
        # 添加最后一个块
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks


# 日期标准化函数
def normalize_date(date_str):
    """
    将不同格式的日期标准化为YYYY-MM-DD格式
    
    Args:
        date_str: 待标准化的日期字符串
        
    Returns:
        str: 标准化后的日期字符串
    """
    # 处理中文日期格式（如2024年6月15日）
    if '年' in date_str and '月' in date_str:
        match = re.match(r'(\d{4})年(\d{1,2})月(\d{1,2})日?', date_str)
        if match:
            year, month, day = match.groups()
            return f"{year}-{int(month):02d}-{int(day):02d}"
    
    # 处理斜杠分隔的日期（如2024/6/15）
    if '/' in date_str:
        parts = date_str.split('/')
        if len(parts) == 3:
            year, month, day = parts
            return f"{year}-{int(month):02d}-{int(day):02d}"
    
    # 处理已经是连字符格式的日期（如2024-6-15）
    if '-' in date_str:
        parts = date_str.split('-')
        if len(parts) == 3:
            year, month, day = parts
            return f"{year}-{int(month):02d}-{int(day):02d}"
    
    # 无法识别的格式，返回原字符串
    return date_str


# 使用示例
if __name__ == "__main__":
    # 创建实体提取器
    extractor = EntityExtractor()  # 不指定模型路径，使用规则匹配
    
    # 测试文本
    test_text = """
    IF2406合约的交割日为2024年6月15日，保证金比例为8%。
    沪铜期货持仓限额为10000手。
    根据规则RULE_2024_001，从2024-01-01起，原油期货的保证金比例调整为15%。
    """
    
    # 提取实体
    entities = extractor.extract_entities(test_text)
    
    # 打印结果
    print("提取的实体:")
    for entity in entities:
        print(f"类型: {entity['type']}, 文本: {entity['text']}, 位置: {entity['start']}-{entity['end']}")
    
    # 日期标准化测试
    print("\n日期标准化测试:")
    test_dates = ["2024年6月15日", "2024/6/15", "2024-6-15"]
    for date in test_dates:
        print(f"{date} -> {normalize_date(date)}")