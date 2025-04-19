import os
import json
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset as HFDataset

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FuturesRelationDataset(torch.utils.data.Dataset):
    """期货领域关系分类数据集"""
    
    def __init__(self, examples, tokenizer, max_length=512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # 编码输入文本
        encoding = self.tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # 转换为单个样本格式
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(example["label"])
        
        return item

class RelationModelTrainer:
    """期货领域关系分类模型训练类"""
    
    def __init__(self, 
                 model_name="bert-base-chinese", 
                 max_length=512,
                 batch_size=8,
                 epochs=3,
                 learning_rate=5e-5,
                 weight_decay=0.01,
                 output_dir="./futures-relation-model"):
        """
        初始化关系分类模型训练器
        
        Args:
            model_name: 预训练模型名称或路径
            max_length: 输入序列最大长度
            batch_size: 训练批次大小
            epochs: 训练轮数
            learning_rate: 学习率
            weight_decay: 权重衰减
            output_dir: 模型输出目录
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.output_dir = output_dir
        
        # 关系类型定义
        self.relation_types = {
            "NO_RELATION": "无关系",
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
        
        # 创建类别映射
        self.label2id = {label: i for i, label in enumerate(self.relation_types.keys())}
        self.id2label = {i: label for i, label in enumerate(self.relation_types.keys())}
        
        # 初始化分词器和模型
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化分词器和模型"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # 确保分词器具有特殊标记
            special_tokens = {
                "additional_special_tokens": [
                    "[SOURCE_START:CONTRACT]", "[SOURCE_END]",
                    "[SOURCE_START:RULE]", 
                    "[SOURCE_START:DATE]", 
                    "[SOURCE_START:MARGIN]", 
                    "[SOURCE_START:POSITION_LIMIT]", 
                    "[SOURCE_START:VARIETY]", 
                    "[SOURCE_START:TERM]", 
                    "[TARGET_START:CONTRACT]", "[TARGET_END]",
                    "[TARGET_START:RULE]", 
                    "[TARGET_START:DATE]", 
                    "[TARGET_START:MARGIN]", 
                    "[TARGET_START:POSITION_LIMIT]", 
                    "[TARGET_START:VARIETY]", 
                    "[TARGET_START:TERM]"
                ]
            }
            
            # 添加特殊标记
            num_added = self.tokenizer.add_special_tokens(special_tokens)
            logger.info(f"添加了 {num_added} 个特殊标记")
            
            # 初始化模型
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=len(self.relation_types),
                id2label=self.id2label,
                label2id=self.label2id
            )
            
            # 调整模型词嵌入大小
            self.model.resize_token_embeddings(len(self.tokenizer))
            
            logger.info(f"模型和分词器初始化成功: {self.model_name}")
        except Exception as e:
            logger.error(f"模型初始化失败: {str(e)}")
            raise e
    
    def prepare_dataset(self, data_path, split_ratio=0.2):
        """
        准备训练和评估数据集
        
        Args:
            data_path: 关系标注数据路径，JSON或CSV格式
            split_ratio: 验证集比例
            
        Returns:
            tuple: (训练集, 验证集)
        """
        try:
            # 加载标注数据
            if data_path.endswith('.json'):
                with open(data_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                
                # 处理JSON格式数据
                if isinstance(raw_data, list):
                    examples = raw_data
                else:
                    logger.error("不支持的JSON数据格式")
                    raise ValueError("不支持的JSON数据格式")
            elif data_path.endswith('.csv'):
                # 假设CSV格式为: text, source_entity, target_entity, relation_type
                df = pd.read_csv(data_path)
                
                examples = []
                for _, row in df.iterrows():
                    # 将实体信息和关系类型转换为字典
                    source_entity = json.loads(row['source_entity']) if isinstance(row['source_entity'], str) else row['source_entity']
                    target_entity = json.loads(row['target_entity']) if isinstance(row['target_entity'], str) else row['target_entity']
                    
                    # 构建带标记的文本
                    marked_text = self._mark_entities_in_text(
                        row['text'], source_entity, target_entity
                    )
                    
                    examples.append({
                        "text": marked_text,
                        "label": self.label2id.get(row['relation_type'], 0)  # 默认为NO_RELATION
                    })
            else:
                logger.error(f"不支持的数据文件格式: {data_path}")
                raise ValueError(f"不支持的数据文件格式: {data_path}")
            
            logger.info(f"加载了 {len(examples)} 条关系标注数据")
            
            # 划分训练集和验证集
            train_examples, val_examples = train_test_split(
                examples, test_size=split_ratio, random_state=42
            )
            
            # 创建数据集
            train_dataset = FuturesRelationDataset(train_examples, self.tokenizer, self.max_length)
            val_dataset = FuturesRelationDataset(val_examples, self.tokenizer, self.max_length)
            
            logger.info(f"准备完成: 训练集 {len(train_dataset)} 条, 验证集 {len(val_dataset)} 条")
            
            return train_dataset, val_dataset
            
        except Exception as e:
            logger.error(f"数据集准备失败: {str(e)}")
            raise e
    
    def _mark_entities_in_text(self, text, source_entity, target_entity):
        """
        在文本中标记实体，用于关系分类
        
        Args:
            text: 原始文本
            source_entity: 源实体字典，包含type、start、end字段
            target_entity: 目标实体字典，包含type、start、end字段
            
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
    
    def train(self, train_dataset, val_dataset=None):
        """
        训练关系分类模型
        
        Args:
            train_dataset: 训练集
            val_dataset: 验证集，可选
            
        Returns:
            训练后的模型
        """
        # 设置训练参数
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            logging_dir=os.path.join(self.output_dir, "logs"),
            logging_steps=100,
            evaluation_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_loss" if val_dataset else None,
            greater_is_better=False,
            fp16=torch.cuda.is_available(),  # 如果有GPU，使用混合精度训练
            report_to="tensorboard"
        )
        
        # 准备训练器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if val_dataset else None
        )
        
        # 开始训练
        logger.info("开始训练模型...")
        train_result = trainer.train()
        
        # 保存最终模型
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        # 输出训练指标
        logger.info(f"训练完成! 训练步数: {train_result.global_step}")
        
        # 评估模型
        if val_dataset:
            logger.info("评估模型...")
            eval_result = trainer.evaluate()
            logger.info(f"评估结果: {eval_result}")
        
        return self.model
    
    def evaluate(self, test_dataset):
        """
        评估模型性能
        
        Args:
            test_dataset: 测试集
            
        Returns:
            dict: 评估指标
        """
        # 加载模型(如果没有训练过)
        if os.path.exists(self.output_dir):
            self.model = AutoModelForSequenceClassification.from_pretrained(self.output_dir)
        
        # 设置评估参数
        eval_args = TrainingArguments(
            output_dir=os.path.join(self.output_dir, "eval"),
            per_device_eval_batch_size=self.batch_size,
            logging_dir=os.path.join(self.output_dir, "eval_logs"),
            report_to="tensorboard"
        )
        
        # 准备评估器
        evaluator = Trainer(
            model=self.model,
            args=eval_args,
            tokenizer=self.tokenizer
        )
        
        # 评估模型
        eval_results = evaluator.evaluate(test_dataset)
        
        # 计算详细指标
        predictions = evaluator.predict(test_dataset)
        preds = np.argmax(predictions.predictions, axis=1)
        labels = predictions.label_ids
        
        # 计算每种关系类型的指标
        metrics = self._compute_relation_metrics(preds, labels)
        
        # 合并指标
        all_metrics = {**eval_results, **metrics}
        
        return all_metrics
    
    def _compute_relation_metrics(self, preds, labels):
        """
        计算每种关系类型的精确率、召回率和F1分数
        
        Args:
            preds: 预测标签序列
            labels: 真实标签序列
            
        Returns:
            dict: 各关系类型的指标
        """
        # 按关系类型统计TP, FP, FN
        stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        
        for pred, label in zip(preds, labels):
            # 跳过NO_RELATION(0)类
            if label != 0:  # 真实标签不是NO_RELATION
                if pred == label:  # 预测正确
                    stats[label]["tp"] += 1
                else:  # 预测错误
                    stats[label]["fn"] += 1
                    if pred != 0:  # 预测为其他关系
                        stats[pred]["fp"] += 1
            elif pred != 0:  # 真实标签是NO_RELATION但预测为其他关系
                stats[pred]["fp"] += 1
        
        # 计算每种关系类型的指标
        metrics = {}
        for relation_id, counts in stats.items():
            tp = counts["tp"]
            fp = counts["fp"]
            fn = counts["fn"]
            
            # 计算精确率
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            # 计算召回率
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            # 计算F1分数
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            relation_name = self.id2label.get(relation_id)
            metrics[f"{relation_name}_precision"] = precision
            metrics[f"{relation_name}_recall"] = recall
            metrics[f"{relation_name}_f1"] = f1
        
        # 计算宏平均F1（不包括NO_RELATION）
        if stats:
            f1_scores = [metrics[f"{self.id2label.get(rel_id)}_f1"] for rel_id in stats.keys() if rel_id != 0]
            if f1_scores:
                metrics["macro_f1"] = sum(f1_scores) / len(f1_scores)
        
        return metrics
    
    def generate_training_examples(self, texts, entity_extractions):
        """
        从带标注实体的文本中生成关系训练数据
        
        Args:
            texts: 文本列表
            entity_extractions: 每个文本对应的实体提取结果列表
            
        Returns:
            list: 关系训练样本列表
        """
        training_examples = []
        
        for text, entities in zip(texts, entity_extractions):
            # 只处理至少有两个实体的文本
            if len(entities) < 2:
                continue
            
            # 生成实体对
            for i in range(len(entities)):
                for j in range(len(entities)):
                    if i == j:
                        continue
                    
                    source_entity = entities[i]
                    target_entity = entities[j]
                    
                    # 标记文本
                    marked_text = self._mark_entities_in_text(text, source_entity, target_entity)
                    
                    # 默认为NO_RELATION，需要人工标注
                    training_examples.append({
                        "text": text,
                        "marked_text": marked_text,
                        "source_entity": source_entity,
                        "target_entity": target_entity,
                        "relation_type": "NO_RELATION"
                    })
        
        return training_examples
    
    def export_training_data(self, training_examples, output_path):
        """
        导出关系训练数据，用于人工标注
        
        Args:
            training_examples: 训练样本列表
            output_path: 输出文件路径
        """
        if output_path.endswith(".json"):
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(training_examples, f, ensure_ascii=False, indent=2)
        elif output_path.endswith(".csv"):
            df = pd.DataFrame(training_examples)
            df.to_csv(output_path, index=False, encoding="utf-8")
        else:
            logger.error(f"不支持的输出文件格式: {output_path}")
            raise ValueError(f"不支持的输出文件格式: {output_path}")
        
        logger.info(f"已导出 {len(training_examples)} 条关系训练数据到 {output_path}")


# 如果直接运行此脚本，执行示例训练
if __name__ == "__main__":
    # 生成示例关系标注数据
    example_data = []
    
    # 示例1：规则适用于合约
    text1 = "根据规则RULE_2024_001，从2024年1月1日起，IF2406合约的保证金比例调整为8%。"
    source_entity1 = {"type": "RULE", "text": "RULE_2024_001", "start": 3, "end": 16}
    target_entity1 = {"type": "CONTRACT", "text": "IF2406", "start": 30, "end": 36}
    
    # 构建带标记的文本
    trainer = RelationModelTrainer()
    marked_text1 = trainer._mark_entities_in_text(text1, source_entity1, target_entity1)
    
    example_data.append({
        "text": marked_text1,
        "label": trainer.label2id["APPLY_TO"]
    })
    
    # 示例2：合约具有保证金比例
    text2 = "IF2406合约的保证金比例为8%。"
    source_entity2 = {"type": "CONTRACT", "text": "IF2406", "start": 0, "end": 6}
    target_entity2 = {"type": "MARGIN", "text": "8%", "start": 14, "end": 16}
    
    marked_text2 = trainer._mark_entities_in_text(text2, source_entity2, target_entity2)
    
    example_data.append({
        "text": marked_text2,
        "label": trainer.label2id["HAS_MARGIN"]
    })
    
    # 示例3：合约的交割日期
    text3 = "IF2406合约的交割日为2024年6月15日。"
    source_entity3 = {"type": "CONTRACT", "text": "IF2406", "start": 0, "end": 6}
    target_entity3 = {"type": "DATE", "text": "2024年6月15日", "start": 12, "end": 23}
    
    marked_text3 = trainer._mark_entities_in_text(text3, source_entity3, target_entity3)
    
    example_data.append({
        "text": marked_text3,
        "label": trainer.label2id["DELIVERY_ON"]
    })
    
    # 保存示例数据
    os.makedirs("data", exist_ok=True)
    with open("data/example_relation_data.json", "w", encoding="utf-8") as f:
        json.dump(example_data, f, ensure_ascii=False, indent=2)
    
    # 创建训练器
    trainer = RelationModelTrainer(
        model_name="bert-base-chinese",  # 使用更小的模型进行测试
        max_length=128,
        batch_size=2,
        epochs=2,
        output_dir="./output/futures-relation-test"
    )
    
    # 准备数据集
    train_dataset, val_dataset = trainer.prepare_dataset("data/example_relation_data.json")
    
    # 训练模型
    trainer.train(train_dataset, val_dataset)
    
    # 评估模型
    metrics = trainer.evaluate(val_dataset)
    print("评估指标:", metrics)