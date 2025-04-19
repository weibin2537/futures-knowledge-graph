import re
import logging
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import spacy
from spacy.tokens import Doc, Span
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    pipeline
)
import torch

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RelationExtractor:
    """期货领域关系提取器"""
    
    def __init__(self, model_path=None, use_spacy=True, use_rules=True, threshold=0.5):
        """
        初始化关系提取器
        
        Args:
            model_path: 关系分类模型路径，如果为None则仅使用规则和依存分析
            use_spacy: 是否使用spaCy进行依存句法分析
            use_rules: 是否使用规则匹配
            threshold: 关系预测阈值
        """
        self.model_path = model_path
        self.use_spacy = use_spacy
        self.use_rules = use_rules
        self.threshold = threshold
        self.model = None
        self.tokenizer = None
        self.nlp = None
        
        # 关系类型定义
        self.relation_types = {
            "APPLY_TO": "规则适用于合约",
            "DEFINED_IN": "在文档中定义",
            "MENTIONED_IN": "在文档中提及",
            "EXPLAINED_IN": "在文档中解释",
            "HAS_MARGIN": "具有保证金比例",
            "HAS_LIMIT": "具有持仓限额",
            "BELONGS_TO": "属于品种",
            "EFFECTIVE_FROM": "从日期生效",
            "DELIVERY_ON": "在日期交割"
        }
        
        # 初始化依赖组件
        self._initialize_components()
    
    def _initialize_components(self):
        """初始化所需组件"""
        # 初始化spaCy
        if self.use_spacy:
            try:
                # 加载中文模型
                self.nlp = spacy.load("zh_core_web_sm")
                logger.info("spaCy模型加载成功")
            except OSError:
                logger.warning("未找到spaCy中文模型，尝试下载...")
                try:
                    # 尝试下载模型
                    spacy.cli.download("zh_core_web_sm")
                    self.nlp = spacy.load("zh_core_web_sm")
                    logger.info("spaCy模型下载并加载成功")
                except Exception as e:
                    logger.error(f"spaCy模型下载失败: {str(e)}")
                    logger.info("使用spaCy的空模型")
                    # 使用空模型
                    self.nlp = spacy.blank("zh")
        
        # 初始化关系分类模型
        if self.model_path:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
                
                logger.info(f"关系分类模型加载成功: {self.model_path}")
            except Exception as e:
                logger.error(f"关系分类模型加载失败: {str(e)}")
                self.model_path = None
    
    def extract_relations(self, text: str, entities: List[Dict]) -> List[Dict]:
        """
        从文本中提取实体间的关系
        
        Args:
            text: 输入文本
            entities: 实体列表，每个实体为一个字典，包含type、text、start、end等字段
            
        Returns:
            list: 关系列表，每个关系为一个字典，包含source、target、type等字段
        """
        if not entities or len(entities) < 2:
            return []
        
        # 对实体按位置排序
        sorted_entities = sorted(entities, key=lambda e: e["start"])
        
        # 存储提取的关系
        relations = []
        
        # 使用规则提取关系
        if self.use_rules:
            rule_relations = self._extract_relations_by_rules(text, sorted_entities)
            relations.extend(rule_relations)
        
        # 使用依存分析提取关系
        if self.use_spacy and self.nlp:
            dependency_relations = self._extract_relations_by_dependency(text, sorted_entities)
            relations.extend(dependency_relations)
        
        # 使用深度学习模型提取关系
        if self.model and self.tokenizer:
            model_relations = self._extract_relations_by_model(text, sorted_entities)
            relations.extend(model_relations)
        
        # 去除重复关系
        unique_relations = self._deduplicate_relations(relations)
        
        return unique_relations
    
    def _extract_relations_by_rules(self, text: str, entities: List[Dict]) -> List[Dict]:
        """
        使用规则提取关系
        
        Args:
            text: 输入文本
            entities: 排序后的实体列表
            
        Returns:
            list: 关系列表
        """
        relations = []
        
        # 按实体类型索引
        entity_by_type = {}
        for entity in entities:
            entity_type = entity["type"]
            if entity_type not in entity_by_type:
                entity_by_type[entity_type] = []
            entity_by_type[entity_type].append(entity)
        
        # 规则1: 规则适用于合约
        if "RULE" in entity_by_type and "CONTRACT" in entity_by_type:
            for rule_entity in entity_by_type["RULE"]:
                rule_position = rule_entity["start"]
                
                for contract_entity in entity_by_type["CONTRACT"]:
                    # 检查文本中是否有表示适用关系的指示词
                    contract_position = contract_entity["start"]
                    
                    # 计算两个实体之间的文本
                    if rule_position < contract_position:
                        middle_text = text[rule_entity["end"]:contract_entity["start"]]
                    else:
                        middle_text = text[contract_entity["end"]:rule_entity["start"]]
                    
                    # 检查指示词
                    if re.search(r'适用|应用于|适用于|对.*有效|执行', middle_text):
                        relations.append({
                            "source": rule_entity["text"],
                            "source_type": "RULE",
                            "target": contract_entity["text"],
                            "target_type": "CONTRACT",
                            "type": "APPLY_TO",
                            "description": self.relation_types["APPLY_TO"],
                            "confidence": 0.9,
                            "extraction_method": "rule"
                        })
        
        # 规则2: 保证金比例关系
        if ("CONTRACT" in entity_by_type or "VARIETY" in entity_by_type) and "MARGIN" in entity_by_type:
            margin_entities = entity_by_type["MARGIN"]
            
            # 合约的保证金比例
            if "CONTRACT" in entity_by_type:
                for contract_entity in entity_by_type["CONTRACT"]:
                    for margin_entity in margin_entities:
                        # 检查两个实体之间的文本是否包含关系指示词
                        contract_position = contract_entity["start"]
                        margin_position = margin_entity["start"]
                        
                        if contract_position < margin_position:
                            middle_text = text[contract_entity["end"]:margin_entity["start"]]
                            if re.search(r'保证金|保证金比例|保证金率', middle_text):
                                relations.append({
                                    "source": contract_entity["text"],
                                    "source_type": "CONTRACT",
                                    "target": margin_entity["text"],
                                    "target_type": "MARGIN",
                                    "type": "HAS_MARGIN",
                                    "description": self.relation_types["HAS_MARGIN"],
                                    "confidence": 0.85,
                                    "extraction_method": "rule"
                                })
            
            # 品种的保证金比例
            if "VARIETY" in entity_by_type:
                for variety_entity in entity_by_type["VARIETY"]:
                    for margin_entity in margin_entities:
                        variety_position = variety_entity["start"]
                        margin_position = margin_entity["start"]
                        
                        if variety_position < margin_position:
                            middle_text = text[variety_entity["end"]:margin_entity["start"]]
                            if re.search(r'保证金|保证金比例|保证金率', middle_text):
                                relations.append({
                                    "source": variety_entity["text"],
                                    "source_type": "VARIETY",
                                    "target": margin_entity["text"],
                                    "target_type": "MARGIN",
                                    "type": "HAS_MARGIN",
                                    "description": self.relation_types["HAS_MARGIN"],
                                    "confidence": 0.85,
                                    "extraction_method": "rule"
                                })
        
        # 规则3: 持仓限额关系
        if ("CONTRACT" in entity_by_type or "VARIETY" in entity_by_type) and "POSITION_LIMIT" in entity_by_type:
            limit_entities = entity_by_type["POSITION_LIMIT"]
            
            # 合约的持仓限额
            if "CONTRACT" in entity_by_type:
                for contract_entity in entity_by_type["CONTRACT"]:
                    for limit_entity in limit_entities:
                        # 检查指示词
                        contract_position = contract_entity["start"]
                        limit_position = limit_entity["start"]
                        
                        if contract_position < limit_position:
                            middle_text = text[contract_entity["end"]:limit_entity["start"]]
                            if re.search(r'持仓限额|限仓|持仓量', middle_text):
                                relations.append({
                                    "source": contract_entity["text"],
                                    "source_type": "CONTRACT",
                                    "target": limit_entity["text"],
                                    "target_type": "POSITION_LIMIT",
                                    "type": "HAS_LIMIT",
                                    "description": self.relation_types["HAS_LIMIT"],
                                    "confidence": 0.85,
                                    "extraction_method": "rule"
                                })
            
            # 品种的持仓限额
            if "VARIETY" in entity_by_type:
                for variety_entity in entity_by_type["VARIETY"]:
                    for limit_entity in limit_entities:
                        variety_position = variety_entity["start"]
                        limit_position = limit_entity["start"]
                        
                        if variety_position < limit_position:
                            middle_text = text[variety_entity["end"]:limit_entity["start"]]
                            if re.search(r'持仓限额|限仓|持仓量', middle_text):
                                relations.append({
                                    "source": variety_entity["text"],
                                    "source_type": "VARIETY",
                                    "target": limit_entity["text"],
                                    "target_type": "POSITION_LIMIT",
                                    "type": "HAS_LIMIT",
                                    "description": self.relation_types["HAS_LIMIT"],
                                    "confidence": 0.85,
                                    "extraction_method": "rule"
                                })
        
        # 规则4: 品种包含合约关系
        if "VARIETY" in entity_by_type and "CONTRACT" in entity_by_type:
            for variety_entity in entity_by_type["VARIETY"]:
                for contract_entity in entity_by_type["CONTRACT"]:
                    # 有些合约代码包含品种信息，如铜期货的CU2406
                    contract_text = contract_entity["text"].lower()
                    variety_text = variety_entity["text"].lower()
                    
                    # 检查合约名称是否包含品种关键词
                    if (variety_text in contract_text or 
                        (variety_text == "沪铜" and "cu" in contract_text) or
                        (variety_text == "沪深300" and "if" in contract_text) or
                        (variety_text == "沪金" and "au" in contract_text)):
                        relations.append({
                            "source": contract_entity["text"],
                            "source_type": "CONTRACT",
                            "target": variety_entity["text"],
                            "target_type": "VARIETY",
                            "type": "BELONGS_TO",
                            "description": self.relation_types["BELONGS_TO"],
                            "confidence": 0.8,
                            "extraction_method": "rule"
                        })
        
        # 规则5: 规则生效日期关系
        if "RULE" in entity_by_type and "DATE" in entity_by_type:
            for rule_entity in entity_by_type["RULE"]:
                for date_entity in entity_by_type["DATE"]:
                    rule_position = rule_entity["start"]
                    date_position = date_entity["start"]
                    
                    # 如果日期在规则之后
                    if rule_position < date_position:
                        middle_text = text[rule_entity["end"]:date_entity["start"]]
                        if re.search(r'(从|自|生效日期|开始生效|生效时间|实施日期)', middle_text):
                            relations.append({
                                "source": rule_entity["text"],
                                "source_type": "RULE",
                                "target": date_entity["text"],
                                "target_type": "DATE",
                                "type": "EFFECTIVE_FROM",
                                "description": self.relation_types["EFFECTIVE_FROM"],
                                "confidence": 0.85,
                                "extraction_method": "rule"
                            })
        
        # 规则6: 合约交割日期关系
        if "CONTRACT" in entity_by_type and "DATE" in entity_by_type:
            for contract_entity in entity_by_type["CONTRACT"]:
                for date_entity in entity_by_type["DATE"]:
                    contract_position = contract_entity["start"]
                    date_position = date_entity["start"]
                    
                    if contract_position < date_position:
                        middle_text = text[contract_entity["end"]:date_entity["start"]]
                        if re.search(r'交割日|交割日期|最后交易日|到期日', middle_text):
                            relations.append({
                                "source": contract_entity["text"],
                                "source_type": "CONTRACT",
                                "target": date_entity["text"],
                                "target_type": "DATE",
                                "type": "DELIVERY_ON",
                                "description": self.relation_types["DELIVERY_ON"],
                                "confidence": 0.9,
                                "extraction_method": "rule"
                            })
        
        return relations
    
    def _extract_relations_by_dependency(self, text: str, entities: List[Dict]) -> List[Dict]:
        """
        使用依存句法分析提取关系
        
        Args:
            text: 输入文本
            entities: 排序后的实体列表
            
        Returns:
            list: 关系列表
        """
        if not self.nlp:
            return []
        
        relations = []
        
        # 使用spaCy处理文本
        doc = self.nlp(text)
        
        # 创建实体跨度字典，将文本位置映射到实体
        entity_spans = {}
        for i, entity in enumerate(entities):
            # 创建spaCy Span对象
            start_char = entity["start"]
            end_char = entity["end"]
            
            # 找到对应的token跨度
            start_token = None
            end_token = None
            for j, token in enumerate(doc):
                if token.idx <= start_char < token.idx + len(token.text):
                    start_token = j
                if token.idx <= end_char <= token.idx + len(token.text):
                    end_token = j
                    break
            
            if start_token is not None and end_token is not None:
                entity_span = doc[start_token:end_token+1]
                entity_spans[(start_char, end_char)] = {
                    "entity": entity,
                    "span": entity_span,
                    "index": i
                }
        
        # 分析句法依存结构，寻找特定模式
        for sent in doc.sents:
            # 提取句子中的实体
            sentence_entities = []
            for key, value in entity_spans.items():
                if sent.start_char <= value["entity"]["start"] < sent.end_char:
                    sentence_entities.append(value)
            
            # 至少需要两个实体才能形成关系
            if len(sentence_entities) < 2:
                continue
            
            # 遍历句子中的所有实体对
            for i in range(len(sentence_entities)):
                for j in range(len(sentence_entities)):
                    if i == j:
                        continue
                    
                    source_entity = sentence_entities[i]["entity"]
                    target_entity = sentence_entities[j]["entity"]
                    source_span = sentence_entities[i]["span"]
                    target_span = sentence_entities[j]["span"]
                    
                    # 检查依存路径
                    path = self._get_dependency_path(source_span.root, target_span.root)
                    if not path:
                        continue
                    
                    # 分析依存路径，提取关系
                    relation = self._analyze_dependency_path(path, source_entity, target_entity)
                    if relation:
                        relations.append(relation)
        
        return relations
    
    def _get_dependency_path(self, source_token, target_token, max_length=5):
        """
        获取两个token之间的最短依存路径
        
        Args:
            source_token: 源token
            target_token: 目标token
            max_length: 最大路径长度
            
        Returns:
            list: 路径上的token列表，如果没有路径或路径过长则返回None
        """
        # 如果源和目标是同一个token
        if source_token == target_token:
            return [source_token]
        
        # 广度优先搜索
        queue = [(source_token, [source_token])]
        visited = {source_token}
        
        while queue:
            current_token, path = queue.pop(0)
            
            # 如果路径已经太长，放弃
            if len(path) > max_length:
                continue
            
            # 遍历当前token的相邻节点（依存关系）
            for child in current_token.children:
                if child == target_token:
                    # 找到目标
                    return path + [child]
                
                if child not in visited:
                    visited.add(child)
                    queue.append((child, path + [child]))
            
            # 检查父节点
            if current_token.head != current_token:  # 非根节点
                if current_token.head == target_token:
                    # 找到目标
                    return path + [current_token.head]
                
                if current_token.head not in visited:
                    visited.add(current_token.head)
                    queue.append((current_token.head, path + [current_token.head]))
        
        # 未找到路径
        return None
    
    def _analyze_dependency_path(self, path, source_entity, target_entity):
        """
        分析依存路径，提取关系
        
        Args:
            path: 依存路径
            source_entity: 源实体
            target_entity: 目标实体
            
        Returns:
            dict: 关系字典，如果没有识别出关系则返回None
        """
        # 提取路径上的关键词和依存关系
        path_text = " ".join([token.text for token in path])
        path_deps = [token.dep_ for token in path]
        
        relation = None
        confidence = 0.7  # 基于依存分析的关系默认置信度
        
        # 根据实体类型和路径特征识别关系
        source_type = source_entity["type"]
        target_type = target_entity["type"]
        
        # 规则-合约关系
        if source_type == "RULE" and target_type == "CONTRACT":
            if re.search(r'适用|应用|执行|有效', path_text):
                relation = {
                    "source": source_entity["text"],
                    "source_type": source_type,
                    "target": target_entity["text"],
                    "target_type": target_type,
                    "type": "APPLY_TO",
                    "description": self.relation_types["APPLY_TO"],
                    "confidence": confidence,
                    "extraction_method": "dependency"
                }
        
        # 合约-保证金关系
        elif source_type == "CONTRACT" and target_type == "MARGIN":
            if re.search(r'保证金|比例', path_text):
                relation = {
                    "source": source_entity["text"],
                    "source_type": source_type,
                    "target": target_entity["text"],
                    "target_type": target_type,
                    "type": "HAS_MARGIN",
                    "description": self.relation_types["HAS_MARGIN"],
                    "confidence": confidence,
                    "extraction_method": "dependency"
                }
        
        # 合约-交割日期关系
        elif source_type == "CONTRACT" and target_type == "DATE":
            if re.search(r'交割|到期|交易日', path_text):
                relation = {
                    "source": source_entity["text"],
                    "source_type": source_type,
                    "target": target_entity["text"],
                    "target_type": target_type,
                    "type": "DELIVERY_ON",
                    "description": self.relation_types["DELIVERY_ON"],
                    "confidence": confidence,
                    "extraction_method": "dependency"
                }
        
        # 规则-生效日期关系
        elif source_type == "RULE" and target_type == "DATE":
            if re.search(r'生效|开始|实施|执行', path_text):
                relation = {
                    "source": source_entity["text"],
                    "source_type": source_type,
                    "target": target_entity["text"],
                    "target_type": target_type,
                    "type": "EFFECTIVE_FROM",
                    "description": self.relation_types["EFFECTIVE_FROM"],
                    "confidence": confidence,
                    "extraction_method": "dependency"
                }
        
        return relation
    
    def _extract_relations_by_model(self, text: str, entities: List[Dict]) -> List[Dict]:
        """
        使用预训练模型提取关系
        
        Args:
            text: 输入文本
            entities: 排序后的实体列表
            
        Returns:
            list: 关系列表
        """
        if not self.model or not self.tokenizer:
            return []
        
        relations = []
        
        # 遍历所有实体对
        for i in range(len(entities)):
            for j in range(len(entities)):
                if i == j:
                    continue
                
                source_entity = entities[i]
                target_entity = entities[j]
                
                # 构建输入文本，使用特殊标记标注实体位置
                marked_text = self._mark_entities_in_text(text, source_entity, target_entity)
                
                # 对输入进行编码
                inputs = self.tokenizer(
                    marked_text,
                    padding="max_length",
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                
                # 预测关系
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=1)
                    
                    # 获取最可能的关系类型
                    max_prob, pred_class = torch.max(probs, dim=1)
                    
                    relation_type = self.model.config.id2label.get(pred_class.item(), "NO_RELATION")
                    confidence = max_prob.item()
                    
                    # 如果不是无关系，且置信度超过阈值
                    if relation_type != "NO_RELATION" and confidence >= self.threshold:
                        # 检查关系类型是否在我们定义的关系类型中
                        if relation_type in self.relation_types:
                            relations.append({
                                "source": source_entity["text"],
                                "source_type": source_entity["type"],
                                "target": target_entity["text"],
                                "target_type": target_entity["type"],
                                "type": relation_type,
                                "description": self.relation_types[relation_type],
                                "confidence": confidence,
                                "extraction_method": "model"
                            })
        
        return relations
    
    def _mark_entities_in_text(self, text: str, source_entity: Dict, target_entity: Dict) -> str:
        """
        在文本中标记实体，用于关系分类模型输入
        
        Args:
            text: 原始文本
            source_entity: 源实体
            target_entity: 目标实体
            
        Returns:
            str: 标记了实体的文本
        """
        # 复制原始文本
        marked_text = text
        
        # 获取实体位置（需要考虑插入标记后的位置偏移）
        source_start = source_entity["start"]
        source_end = source_entity["end"]
        target_start = target_entity["start"]
        target_end = target_entity["end"]
        
        # 调整位置，确保先处理后面的实体，避免位置错乱
        if source_start > target_start:
            # 先标记目标实体
            marked_text = (marked_text[:target_start] + 
                          f"[TARGET_START:{target_entity['type']}]" + 
                          marked_text[target_start:target_end] + 
                          "[TARGET_END]" + 
                          marked_text[target_end:])
            
            # 调整源实体位置
            offset = len(f"[TARGET_START:{target_entity['type']}]") + len("[TARGET_END]")
            source_start += offset if source_start > target_end else (len(f"[TARGET_START:{target_entity['type']}]") if source_start > target_start else 0)
            source_end += offset if source_end > target_end else (len(f"[TARGET_START:{target_entity['type']}]") if source_end > target_start else 0)
            
            # 标记源实体
            marked_text = (marked_text[:source_start] + 
                          f"[SOURCE_START:{source_entity['type']}]" + 
                          marked_text[source_start:source_end] + 
                          "[SOURCE_END]" + 
                          marked_text[source_end:])
        else:
            # 先标记源实体
            marked_text = (marked_text[:source_start] + 
                          f"[SOURCE_START:{source_entity['type']}]" + 
                          marked_text[source_start:source_end] + 
                          "[SOURCE_END]" + 
                          marked_text[source_end:])
            
            # 调整目标实体位置
            offset = len(f"[SOURCE_START:{source_entity['type']}]") + len("[SOURCE_END]")
            target_start += offset if target_start > source_end else (len(f"[SOURCE_START:{source_entity['type']}]") if target_start > source_start else 0)
            target_end += offset if target_end > source_end else (len(f"[SOURCE_START:{source_entity['type']}]") if target_end > source_start else 0)
            
            # 标记目标实体
            marked_text = (marked_text[:target_start] + 
                          f"[TARGET_START:{target_entity['type']}]" + 
                          marked_text[target_start:target_end] + 
                          "[TARGET_END]" + 
                          marked_text[target_end:])
        
        return marked_text
    
    def _deduplicate_relations(self, relations: List[Dict]) -> List[Dict]:
        """
        去除重复关系，保留置信度最高的
        
        Args:
            relations: 关系列表
            
        Returns:
            list: 去重后的关系列表
        """
        if not relations:
            return []
        
        # 按关系的关键属性分组
        relation_map = {}
        for relation in relations:
            key = (relation["source"], relation["source_type"], 
                   relation["target"], relation["target_type"], 
                   relation["type"])
            
            # 如果已经存在该关系，保留置信度更高的
            if key in relation_map:
                if relation["confidence"] > relation_map[key]["confidence"]:
                    relation_map[key] = relation
            else:
                relation_map[key] = relation
        
        # 返回去重后的关系列表
        return list(relation_map.values())


# 示例使用
if __name__ == "__main__":
    # 创建关系提取器
    relation_extractor = RelationExtractor()
    
    # 示例文本
    test_text = """
    根据规则RULE_2024_001，从2024年1月1日起，IF2406合约的保证金比例调整为8%，交割日为2024年6月15日。
    沪铜期货持仓限额为10000手。
    """
    
    # 示例实体
    test_entities = [
        {"type": "RULE", "text": "RULE_2024_001", "start": 3, "end": 16},
        {"type": "DATE", "text": "2024年1月1日", "start": 18, "end": 28},
        {"type": "CONTRACT", "text": "IF2406", "start": 30, "end": 36},
        {"type": "MARGIN", "text": "8%", "start": 44, "end": 46},
        {"type": "DATE", "text": "2024年6月15日", "start": 52, "end": 63},
        {"type": "VARIETY", "text": "沪铜", "start": 65, "end": 67},
        {"type": "POSITION_LIMIT", "text": "10000", "start": 73, "end": 78}
    ]
    
    # 提取关系
    relations = relation_extractor.extract_relations(test_text, test_entities)
    
    # 打印结果
    print(f"提取到 {len(relations)} 条关系:")
    for relation in relations:
        print(f"{relation['source']} ({relation['source_type']}) --[{relation['type']}]--> {relation['target']} ({relation['target_type']})")
        print(f"  描述: {relation['description']}")
        print(f"  置信度: {relation['confidence']:.2f}")
        print(f"  提取方法: {relation['extraction_method']}")
        print()