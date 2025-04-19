import os
import json
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForTokenClassification, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
from datasets import Dataset as HFDataset

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FuturesNERTrainer:
    """期货领域命名实体识别模型训练类"""
    
    def __init__(self, 
                 model_name="deepseek-ai/deepseek-llm-7b-base", 
                 max_length=512,
                 batch_size=8,
                 epochs=3,
                 learning_rate=5e-5,
                 weight_decay=0.01,
                 output_dir="./futures-ner-model"):
        """
        初始化NER训练器
        
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
        
        # 实体标签映射
        self.label_list = [
            "O",  # 非实体
            "B-CONTRACT", "I-CONTRACT",  # 合约
            "B-RULE", "I-RULE",  # 规则
            "B-DATE", "I-DATE",  # 日期
            "B-MARGIN", "I-MARGIN",  # 保证金比例
            "B-POSITION_LIMIT", "I-POSITION_LIMIT",  # 持仓限额
            "B-VARIETY", "I-VARIETY",  # 品种
            "B-TERM", "I-TERM"  # 术语
        ]
        self.label2id = {label: i for i, label in enumerate(self.label_list)}
        self.id2label = {i: label for i, label in enumerate(self.label_list)}
        
        # 初始化分词器和模型
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化分词器和模型"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # 确保分词器具有特殊标记
            special_tokens = {
                "pad_token": "[PAD]",
                "unk_token": "[UNK]",
                "sep_token": "[SEP]",
                "cls_token": "[CLS]"
            }
            # 为可能缺少的特殊标记添加
            added_tokens = []
            for token_name, token_value in special_tokens.items():
                if getattr(self.tokenizer, token_name) is None:
                    added_tokens.append(token_value)
            
            if added_tokens:
                self.tokenizer.add_special_tokens({"additional_special_tokens": added_tokens})
                for token_name, token_value in special_tokens.items():
                    if getattr(self.tokenizer, token_name) is None:
                        setattr(self.tokenizer, token_name, token_value)
            
            # 初始化模型
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.model_name,
                num_labels=len(self.label_list),
                id2label=self.id2label,
                label2id=self.label2id
            )
            
            # 如果添加了特殊标记，调整模型的词嵌入大小
            if added_tokens:
                self.model.resize_token_embeddings(len(self.tokenizer))
                
            logger.info(f"模型和分词器初始化成功: {self.model_name}")
        except Exception as e:
            logger.error(f"模型初始化失败: {str(e)}")
            raise e
    
    def prepare_dataset(self, data_path, split_ratio=0.8):
        """
        准备训练和评估数据集
        
        Args:
            data_path: 标注数据路径，支持LabelStudio导出的JSON格式
            split_ratio: 训练集比例
            
        Returns:
            tuple: (训练集, 评估集)
        """
        try:
            # 加载标注数据
            if data_path.endswith('.json'):
                with open(data_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                
                # 处理LabelStudio格式数据
                if isinstance(raw_data, list) and all(['text' in item and 'entities' in item for item in raw_data]):
                    examples = raw_data
                else:
                    logger.error("不支持的JSON数据格式")
                    raise ValueError("不支持的JSON数据格式")
            elif data_path.endswith('.csv'):
                # 假设CSV格式为: text, entities (JSON字符串)
                df = pd.read_csv(data_path)
                examples = []
                for _, row in df.iterrows():
                    examples.append({
                        'text': row['text'],
                        'entities': json.loads(row['entities'])
                    })
            else:
                logger.error(f"不支持的数据文件格式: {data_path}")
                raise ValueError(f"不支持的数据文件格式: {data_path}")
            
            logger.info(f"加载了 {len(examples)} 条标注数据")
            
            # 将数据转换为模型所需格式
            tokenized_examples = []
            for example in tqdm(examples, desc="处理标注数据"):
                text = example['text']
                entities = example['entities']
                
                # 对文本进行分词
                tokens = self.tokenizer.tokenize(text)
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                
                # 初始化标签序列(全部为O)
                labels = ["O"] * len(tokens)
                
                # 标记实体
                for entity in entities:
                    if 'start' in entity and 'end' in entity and 'label' in entity:
                        start_char = entity['start']
                        end_char = entity['end']
                        entity_type = entity['label']
                        
                        # 查找起始和结束token索引
                        token_start, token_end = self._char_to_token_positions(
                            text, tokens, start_char, end_char
                        )
                        
                        if token_start is not None and token_end is not None:
                            # BIO标注: 第一个token标为B-Type, 其余标为I-Type
                            labels[token_start] = f"B-{entity_type}"
                            for i in range(token_start + 1, token_end + 1):
                                labels[i] = f"I-{entity_type}"
                
                # 转换为ID
                label_ids = [self.label2id.get(label, 0) for label in labels]
                
                # 添加到示例列表
                tokenized_examples.append({
                    "input_ids": token_ids,
                    "labels": label_ids,
                    "text": text,
                    "tokens": tokens
                })
            
            # 将数据集转换为HuggingFace Dataset格式
            dataset = HFDataset.from_list(tokenized_examples)
            
            # 划分训练集和验证集
            train_test_split = dataset.train_test_split(test_size=(1-split_ratio))
            train_dataset = train_test_split["train"]
            eval_dataset = train_test_split["test"]
            
            logger.info(f"准备完成: 训练集 {len(train_dataset)} 条, 验证集 {len(eval_dataset)} 条")
            
            return train_dataset, eval_dataset
        
        except Exception as e:
            logger.error(f"数据集准备失败: {str(e)}")
            raise e
    
    def _char_to_token_positions(self, text, tokens, start_char, end_char):
        """
        将字符位置转换为token位置
        
        Args:
            text: 原始文本
            tokens: 分词后的tokens
            start_char: 实体起始字符位置
            end_char: 实体结束字符位置
            
        Returns:
            tuple: (起始token索引, 结束token索引)
        """
        # 这是一个简化的实现，真实情况更复杂，需要考虑分词器的特性
        current_pos = 0
        token_start = None
        token_end = None
        
        for i, token in enumerate(tokens):
            # 移除可能的特殊字符前缀(如##)
            clean_token = token.replace("##", "")
            if token.startswith("##"):
                # 如果是wordpiece分词的一部分，不增加current_pos
                pass
            else:
                # 找到token在原文中的位置
                while current_pos < len(text) and text[current_pos].isspace():
                    current_pos += 1
            
            token_len = len(clean_token)
            
            # 检查token是否覆盖了实体起始位置
            if token_start is None and current_pos <= start_char < current_pos + token_len:
                token_start = i
            
            # 检查token是否覆盖了实体结束位置
            if token_end is None and current_pos <= end_char <= current_pos + token_len:
                token_end = i
                break
            
            current_pos += token_len
        
        return token_start, token_end
    
    def train(self, train_dataset, eval_dataset=None):
        """
        训练NER模型
        
        Args:
            train_dataset: 训练集
            eval_dataset: 评估集，可选
            
        Returns:
            训练后的模型
        """
        # 创建数据整理器
        data_collator = DataCollatorForTokenClassification(
            self.tokenizer,
            pad_to_multiple_of=8 if torch.cuda.is_available() else None
        )
        
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
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False,
            fp16=torch.cuda.is_available(),  # 如果有GPU，使用混合精度训练
            report_to="tensorboard"
        )
        
        # 准备训练器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if eval_dataset else None
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
        if eval_dataset:
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
            self.model = AutoModelForTokenClassification.from_pretrained(self.output_dir)
        
        # 创建数据整理器
        data_collator = DataCollatorForTokenClassification(
            self.tokenizer,
            pad_to_multiple_of=8 if torch.cuda.is_available() else None
        )
        
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
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # 评估模型
        eval_results = evaluator.evaluate(test_dataset)
        
        # 计算每种实体类型的F1分数
        predictions = evaluator.predict(test_dataset)
        preds = np.argmax(predictions.predictions, axis=2)
        labels = predictions.label_ids
        
        entity_metrics = self._compute_entity_metrics(preds, labels)
        
        # 合并指标
        all_metrics = {**eval_results, **entity_metrics}
        
        return all_metrics
    
    def _compute_entity_metrics(self, preds, labels):
        """
        计算每种实体类型的精确率、召回率和F1分数
        
        Args:
            preds: 预测标签ID序列
            labels: 真实标签ID序列
            
        Returns:
            dict: 各实体类型的指标
        """
        # 按实体类型统计TP, FP, FN
        stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        
        for pred_seq, label_seq in zip(preds, labels):
            # 跳过padding标记(-100)
            valid_indices = label_seq != -100
            if not any(valid_indices):
                continue
                
            pred_seq = pred_seq[valid_indices]
            label_seq = label_seq[valid_indices]
            
            # 提取预测的实体
            pred_entities = self._extract_entities(pred_seq)
            # 提取真实实体
            true_entities = self._extract_entities(label_seq)
            
            # 按实体类型统计
            for entity_type in set(type for type, _, _ in pred_entities + true_entities):
                # 该类型的预测实体
                pred_entities_of_type = {(s, e) for t, s, e in pred_entities if t == entity_type}
                # 该类型的真实实体
                true_entities_of_type = {(s, e) for t, s, e in true_entities if t == entity_type}
                
                # 计算TP, FP, FN
                tp = len(pred_entities_of_type & true_entities_of_type)
                fp = len(pred_entities_of_type - true_entities_of_type)
                fn = len(true_entities_of_type - pred_entities_of_type)
                
                # 累加统计
                stats[entity_type]["tp"] += tp
                stats[entity_type]["fp"] += fp
                stats[entity_type]["fn"] += fn
        
        # 计算每种实体类型的指标
        metrics = {}
        for entity_type, counts in stats.items():
            tp = counts["tp"]
            fp = counts["fp"]
            fn = counts["fn"]
            
            # 计算精确率
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            # 计算召回率
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            # 计算F1分数
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            entity_name = self.id2label.get(entity_type, str(entity_type)).split('-')[-1]
            metrics[f"{entity_name}_precision"] = precision
            metrics[f"{entity_name}_recall"] = recall
            metrics[f"{entity_name}_f1"] = f1
        
        # 计算宏平均F1
        if stats:
            f1_scores = [metrics[f"{self.id2label.get(ent, str(ent)).split('-')[-1]}_f1"] for ent in stats.keys()]
            metrics["macro_f1"] = sum(f1_scores) / len(f1_scores)
        
        return metrics
    
    def _extract_entities(self, label_seq):
        """
        从标签序列中提取实体
        
        Args:
            label_seq: 标签ID序列
            
        Returns:
            list: [(实体类型, 起始位置, 结束位置)]
        """
        entities = []
        current_entity = None
        
        for i, label_id in enumerate(label_seq):
            label = self.id2label.get(label_id, "O")
            
            # B标签表示实体开始
            if label.startswith("B-"):
                # 如果有未结束的实体，先保存
                if current_entity:
                    entities.append(current_entity)
                
                # 开始新实体
                entity_type = label_id
                current_entity = (entity_type, i, i)
            
            # I标签表示实体继续
            elif label.startswith("I-") and current_entity:
                entity_type = label_id
                # 检查是否与当前实体类型相同
                if entity_type == current_entity[0] or entity_type - 1 == current_entity[0]:
                    # 更新实体结束位置
                    current_entity = (current_entity[0], current_entity[1], i)
            
            # O标签或其他情况，结束当前实体
            elif current_entity:
                entities.append(current_entity)
                current_entity = None
        
        # 添加最后一个实体
        if current_entity:
            entities.append(current_entity)
        
        return entities

# 如果直接运行此脚本，执行示例训练
if __name__ == "__main__":
    # 示例标注数据
    example_data = [
        {
            "text": "IF2406合约的交割日为2024年6月15日，保证金比例为8%。",
            "entities": [
                {"start": 0, "end": 6, "label": "CONTRACT"},
                {"start": 11, "end": 19, "label": "DATE"},
                {"start": 25, "end": 27, "label": "MARGIN"}
            ]
        },
        {
            "text": "沪铜期货持仓限额为10000手。",
            "entities": [
                {"start": 0, "end": 2, "label": "VARIETY"},
                {"start": 8, "end": 13, "label": "POSITION_LIMIT"}
            ]
        },
        {
            "text": "根据规则RULE_2024_001，从2024-01-01起，原油期货的保证金比例调整为15%。",
            "entities": [
                {"start": 3, "end": 16, "label": "RULE"},
                {"start": 18, "end": 28, "label": "DATE"},
                {"start": 30, "end": 32, "label": "VARIETY"},
                {"start": 39, "end": 42, "label": "MARGIN"}
            ]
        }
    ]
    
    # 保存示例数据
    os.makedirs("data", exist_ok=True)
    with open("data/example_data.json", "w", encoding="utf-8") as f:
        json.dump(example_data, f, ensure_ascii=False, indent=2)
    
    # 创建训练器
    trainer = FuturesNERTrainer(
        model_name="bert-base-chinese",  # 使用更小的模型进行测试
        max_length=128,
        batch_size=2,
        epochs=2,
        output_dir="./output/futures-ner-test"
    )
    
    # 准备数据集
    train_dataset, eval_dataset = trainer.prepare_dataset("data/example_data.json")
    
    # 训练模型
    trainer.train(train_dataset, eval_dataset)
    
    # 评估模型
    metrics = trainer.evaluate(eval_dataset)
    print("评估指标:", metrics)