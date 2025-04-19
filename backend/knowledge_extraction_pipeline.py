import os
import json
import logging
from typing import Dict, List, Any, Optional, Union
import pandas as pd
from datetime import datetime

# 导入自定义模块
from document_parser import ParserFactory
from entity_extraction import EntityExtractor
from relation_extraction import RelationExtractor
from knowledge_graph import KnowledgeGraph

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KnowledgeExtractionPipeline:
    """期货业务知识提取与图谱构建流水线"""
    
    def __init__(self, 
                 neo4j_uri: str, 
                 neo4j_user: str, 
                 neo4j_password: str,
                 entity_model_path: Optional[str] = None,
                 relation_model_path: Optional[str] = None,
                 output_dir: str = "./output"):
        """
        初始化知识提取流水线
        
        Args:
            neo4j_uri: Neo4j数据库URI
            neo4j_user: Neo4j用户名
            neo4j_password: Neo4j密码
            entity_model_path: 实体识别模型路径，可选
            relation_model_path: 关系分类模型路径，可选
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化各组件
        self.entity_extractor = EntityExtractor(model_path=entity_model_path)
        self.relation_extractor = RelationExtractor(model_path=relation_model_path)
        self.knowledge_graph = KnowledgeGraph(neo4j_uri, neo4j_user, neo4j_password)
        
        logger.info("知识提取与图谱构建流水线初始化完成")
    
    def process_document(self, 
                         file_path: str, 
                         doc_id: Optional[str] = None) -> Dict[str, Any]:
        """
        处理文档，提取实体和关系，构建知识图谱
        
        Args:
            file_path: 文档路径
            doc_id: 文档ID，如果为None则自动创建
            
        Returns:
            dict: 处理结果，包含文档ID、元数据、提取的实体和关系等
        """
        try:
            # 1. 解析文档
            logger.info(f"开始处理文档: {file_path}")
            parser = ParserFactory.create_parser(file_path)
            parse_result = parser.parse(file_path)
            
            if "error" in parse_result:
                logger.error(f"文档解析失败: {parse_result['error']}")
                return {"error": parse_result["error"]}
            
            # 2. 创建文档节点（如果没有提供doc_id）
            if not doc_id:
                doc_id = self.knowledge_graph.create_document_node(parse_result["meta"])
            
            # 3. 提取实体
            text = parse_result.get("text", "")
            entities = []
            
            if text:
                entities = self.entity_extractor.extract_entities(text)
                logger.info(f"提取到 {len(entities)} 个实体")
            
            # 4. 提取关系
            relations = []
            if len(entities) >= 2:
                relations = self.relation_extractor.extract_relations(text, entities)
                logger.info(f"提取到 {len(relations)} 条关系")
            
            # 5. 构建知识图谱
            entity_nodes = self._create_entity_nodes(entities, doc_id)
            relation_edges = self._create_relation_edges(relations)
            
            # 6. 保存处理结果
            result = {
                "doc_id": doc_id,
                "metadata": parse_result["meta"],
                "entities": entities,
                "relations": relations,
                "entity_nodes": entity_nodes,
                "relation_edges": relation_edges,
                "processed_at": datetime.now().isoformat()
            }
            
            # 保存到文件
            result_file = os.path.join(
                self.output_dir, 
                f"doc_{os.path.basename(file_path).split('.')[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            logger.info(f"文档处理完成，结果保存至: {result_file}")
            
            return result
        
        except Exception as e:
            logger.error(f"文档处理失败: {str(e)}")
            return {"error": str(e)}
    
    def process_text(self, 
                     text: str, 
                     metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        处理文本，提取实体和关系，构建知识图谱
        
        Args:
            text: 输入文本
            metadata: 文本元数据，可选
            
        Returns:
            dict: 处理结果，包含提取的实体和关系等
        """
        try:
            logger.info(f"开始处理文本，长度: {len(text)}")
            
            # 1. 创建文档节点（如果提供了元数据）
            doc_id = None
            if metadata:
                doc_id = self.knowledge_graph.create_document_node(metadata)
            
            # 2. 提取实体
            entities = self.entity_extractor.extract_entities(text)
            logger.info(f"提取到 {len(entities)} 个实体")
            
            # 3. 提取关系
            relations = []
            if len(entities) >= 2:
                relations = self.relation_extractor.extract_relations(text, entities)
                logger.info(f"提取到 {len(relations)} 条关系")
            
            # 4. 构建知识图谱
            entity_nodes = self._create_entity_nodes(entities, doc_id)
            relation_edges = self._create_relation_edges(relations)
            
            # 5. 返回处理结果
            result = {
                "doc_id": doc_id,
                "metadata": metadata,
                "text": text,
                "entities": entities,
                "relations": relations,
                "entity_nodes": entity_nodes,
                "relation_edges": relation_edges,
                "processed_at": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"文本处理失败: {str(e)}")
            return {"error": str(e)}
    
    def batch_process(self, 
                      file_paths: List[str], 
                      output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        批量处理文档
        
        Args:
            file_paths: 文档路径列表
            output_file: 输出文件路径，可选
            
        Returns:
            dict: 批处理结果，包含成功和失败的统计
        """
        results = {
            "total": len(file_paths),
            "success": 0,
            "failure": 0,
            "doc_ids": [],
            "errors": {},
            "start_time": datetime.now().isoformat(),
        }
        
        logger.info(f"开始批量处理 {len(file_paths)} 个文档")
        
        for file_path in file_paths:
            try:
                result = self.process_document(file_path)
                
                if "error" in result:
                    results["failure"] += 1
                    results["errors"][file_path] = result["error"]
                else:
                    results["success"] += 1
                    results["doc_ids"].append(result["doc_id"])
            
            except Exception as e:
                results["failure"] += 1
                results["errors"][file_path] = str(e)
        
        results["end_time"] = datetime.now().isoformat()
        
        # 保存批处理结果
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"批处理完成。成功: {results['success']}, 失败: {results['failure']}")
        
        return results
    
    def extract_knowledge_from_directory(self, 
                                         directory: str, 
                                         recursive: bool = True,
                                         file_extensions: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        从目录中提取知识
        
        Args:
            directory: 目录路径
            recursive: 是否递归处理子目录
            file_extensions: 要处理的文件扩展名列表，例如['.pdf', '.docx']，默认处理所有支持的格式
            
        Returns:
            dict: 处理结果统计
        """
        if file_extensions is None:
            file_extensions = ['.pdf', '.docx', '.doc', '.xlsx', '.xls', '.md', '.markdown']
        
        file_paths = []
        
        # 收集符合条件的文件
        if recursive:
            for root, _, files in os.walk(directory):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in file_extensions):
                        file_paths.append(os.path.join(root, file))
        else:
            for file in os.listdir(directory):
                if os.path.isfile(os.path.join(directory, file)) and any(file.lower().endswith(ext) for ext in file_extensions):
                    file_paths.append(os.path.join(directory, file))
        
        logger.info(f"在目录 {directory} 中找到 {len(file_paths)} 个文件需要处理")
        
        # 批量处理文件
        output_file = os.path.join(self.output_dir, f"directory_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        return self.batch_process(file_paths, output_file)
    
    def export_knowledge_graph(self, 
                              output_format: str = "json", 
                              query: Optional[str] = None,
                              output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        导出知识图谱数据
        
        Args:
            output_format: 输出格式，支持json、csv、cypher
            query: 自定义Cypher查询，可选
            output_file: 输出文件路径，可选
            
        Returns:
            dict: 导出结果
        """
        if query is None:
            # 默认查询：提取所有实体和关系
            query = """
            MATCH (n)
            OPTIONAL MATCH (n)-[r]->(m)
            RETURN n, r, m
            """
        
        try:
            # 执行查询
            results = self.knowledge_graph.execute_cypher(query)
            
            # 根据输出格式处理结果
            if output_format == "json":
                output_data = {
                    "nodes": [],
                    "edges": []
                }
                
                # 处理节点和边
                node_ids = set()
                for record in results:
                    # 处理起始节点
                    if "n" in record and record["n"] is not None:
                        node = record["n"]
                        node_id = next((value for key, value in node.items() if key.endswith("_id")), None)
                        
                        if node_id and node_id not in node_ids:
                            node_ids.add(node_id)
                            output_data["nodes"].append(node)
                    
                    # 处理目标节点
                    if "m" in record and record["m"] is not None:
                        node = record["m"]
                        node_id = next((value for key, value in node.items() if key.endswith("_id")), None)
                        
                        if node_id and node_id not in node_ids:
                            node_ids.add(node_id)
                            output_data["nodes"].append(node)
                    
                    # 处理关系
                    if "r" in record and record["r"] is not None:
                        edge = record["r"]
                        output_data["edges"].append(edge)
                
                # 保存到文件
                if output_file:
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(output_data, f, ensure_ascii=False, indent=2)
                
                return {
                    "format": "json",
                    "node_count": len(output_data["nodes"]),
                    "edge_count": len(output_data["edges"]),
                    "output_file": output_file
                }
            
            elif output_format == "csv":
                # 提取节点
                nodes_df = pd.DataFrame()
                edges_df = pd.DataFrame()
                
                for record in results:
                    # 处理节点
                    for key in ["n", "m"]:
                        if key in record and record[key] is not None:
                            node = record[key]
                            node_df = pd.DataFrame([node])
                            nodes_df = pd.concat([nodes_df, node_df], ignore_index=True)
                    
                    # 处理关系
                    if "r" in record and record["r"] is not None:
                        edge = record["r"]
                        edge_df = pd.DataFrame([edge])
                        edges_df = pd.concat([edges_df, edge_df], ignore_index=True)
                
                # 去重
                if not nodes_df.empty:
                    id_columns = [col for col in nodes_df.columns if col.endswith("_id")]
                    if id_columns:
                        nodes_df = nodes_df.drop_duplicates(subset=id_columns[0])
                
                # 保存到文件
                if output_file:
                    nodes_file = output_file.replace(".csv", "_nodes.csv")
                    edges_file = output_file.replace(".csv", "_edges.csv")
                    
                    if not nodes_df.empty:
                        nodes_df.to_csv(nodes_file, index=False, encoding="utf-8")
                    
                    if not edges_df.empty:
                        edges_df.to_csv(edges_file, index=False, encoding="utf-8")
                
                return {
                    "format": "csv",
                    "node_count": len(nodes_df) if not nodes_df.empty else 0,
                    "edge_count": len(edges_df) if not edges_df.empty else 0,
                    "nodes_file": nodes_file if not nodes_df.empty else None,
                    "edges_file": edges_file if not edges_df.empty else None
                }
            
            elif output_format == "cypher":
                # 生成Cypher创建语句
                cypher_statements = []
                
                # 用于跟踪已处理的节点和关系
                processed_nodes = set()
                processed_edges = set()
                
                for record in results:
                    # 处理起始节点
                    if "n" in record and record["n"] is not None:
                        node = record["n"]
                        node_id = next((value for key, value in node.items() if key.endswith("_id")), None)
                        node_label = next((key.split(".")[0] for key in node.keys() if "." in key), "Node")
                        
                        if node_id and node_id not in processed_nodes:
                            processed_nodes.add(node_id)
                            
                            # 构建节点属性
                            props = ", ".join([f"{k}: {self._format_cypher_value(v)}" for k, v in node.items()])
                            
                            # 生成创建节点的Cypher语句
                            cypher_statements.append(f"CREATE (:{node_label} {{{props}}})")
                    
                    # 处理目标节点
                    if "m" in record and record["m"] is not None:
                        node = record["m"]
                        node_id = next((value for key, value in node.items() if key.endswith("_id")), None)
                        node_label = next((key.split(".")[0] for key in node.keys() if "." in key), "Node")
                        
                        if node_id and node_id not in processed_nodes:
                            processed_nodes.add(node_id)
                            
                            # 构建节点属性
                            props = ", ".join([f"{k}: {self._format_cypher_value(v)}" for k, v in node.items()])
                            
                            # 生成创建节点的Cypher语句
                            cypher_statements.append(f"CREATE (:{node_label} {{{props}}})")
                    
                    # 处理关系
                    if all(key in record and record[key] is not None for key in ["n", "r", "m"]):
                        source_node = record["n"]
                        target_node = record["m"]
                        relation = record["r"]
                        
                        source_id = next((value for key, value in source_node.items() if key.endswith("_id")), None)
                        target_id = next((value for key, value in target_node.items() if key.endswith("_id")), None)
                        relation_type = relation.get("type", "RELATED_TO")
                        
                        edge_key = f"{source_id}-{relation_type}-{target_id}"
                        
                        if source_id and target_id and edge_key not in processed_edges:
                            processed_edges.add(edge_key)
                            
                            source_label = next((key.split(".")[0] for key in source_node.keys() if "." in key), "Node")
                            target_label = next((key.split(".")[0] for key in target_node.keys() if "." in key), "Node")
                            
                            source_id_key = next((key for key in source_node.keys() if key.endswith("_id")), None)
                            target_id_key = next((key for key in target_node.keys() if key.endswith("_id")), None)
                            
                            # 生成创建关系的Cypher语句
                            relation_props = ", ".join([f"{k}: {self._format_cypher_value(v)}" for k, v in relation.items() if k != "type"])
                            
                            cypher = f"""
                            MATCH (a:{source_label}), (b:{target_label})
                            WHERE a.{source_id_key} = {self._format_cypher_value(source_id)} AND b.{target_id_key} = {self._format_cypher_value(target_id)}
                            CREATE (a)-[:{relation_type} {{{relation_props}}}]->(b)
                            """
                            
                            cypher_statements.append(cypher.strip())
                
                # 保存到文件
                if output_file:
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write("// 期货业务知识图谱导出脚本\n")
                        f.write(f"// 生成时间: {datetime.now().isoformat()}\n\n")
                        
                        for statement in cypher_statements:
                            f.write(f"{statement};\n\n")
                
                return {
                    "format": "cypher",
                    "statement_count": len(cypher_statements),
                    "output_file": output_file
                }
            
            else:
                return {"error": f"不支持的输出格式: {output_format}"}
        
        except Exception as e:
            logger.error(f"导出知识图谱失败: {str(e)}")
            return {"error": str(e)}
    
    def _create_entity_nodes(self, entities, doc_id=None):
        """创建实体节点并返回结果"""
        entity_nodes = []
        
        for entity in entities:
            entity_type = entity["type"]
            entity_text = entity["text"]
            
            try:
                # 根据实体类型创建不同类型的节点
                if entity_type == "CONTRACT":
                    # 创建合约节点
                    contract_data = {
                        "contract_id": entity_text,
                        "variety": "",  # 需要从关系中提取
                        "delivery_date": ""  # 需要从关系中提取
                    }
                    node_id = self.knowledge_graph.create_contract_node(contract_data, doc_id)
                    entity_nodes.append({
                        "id": node_id,
                        "type": entity_type,
                        "text": entity_text
                    })
                
                elif entity_type == "RULE":
                    # 创建规则节点
                    rule_data = {
                        "rule_id": entity_text,
                        "effective_date": "",  # 需要从关系中提取
                        "description": ""  # 需要从上下文中提取
                    }
                    node_id = self.knowledge_graph.create_rule_node(rule_data, doc_id)
                    entity_nodes.append({
                        "id": node_id,
                        "type": entity_type,
                        "text": entity_text
                    })
                
                elif entity_type == "VARIETY":
                    # 创建品种节点
                    node_id = self.knowledge_graph.create_variety_node(entity_text, doc_id)
                    entity_nodes.append({
                        "id": node_id,
                        "type": entity_type,
                        "text": entity_text
                    })
                
                elif entity_type == "TERM":
                    # 创建术语节点
                    term_data = {
                        "term_name": entity_text,
                        "definition": ""  # 需要从上下文中提取
                    }
                    node_id = self.knowledge_graph.create_term_node(term_data, doc_id)
                    entity_nodes.append({
                        "id": node_id,
                        "type": entity_type,
                        "text": entity_text
                    })
                
                # 其他类型的实体不需要创建节点，但可能用于建立关系
                # 例如日期、保证金比例等可以作为节点属性
            
            except Exception as e:
                logger.error(f"创建实体节点失败: {entity_type} {entity_text} - {str(e)}")
        
        return entity_nodes
    
    def _create_relation_edges(self, relations):
        """创建关系边并返回结果"""
        relation_edges = []
        
        for relation in relations:
            try:
                source_type = relation["source_type"]
                target_type = relation["target_type"]
                source_text = relation["source"]
                target_text = relation["target"]
                relation_type = relation["type"]
                
                # 创建规则适用于合约的关系
                if relation_type == "APPLY_TO" and source_type == "RULE" and target_type == "CONTRACT":
                    success = self.knowledge_graph.create_apply_relation(source_text, target_text)
                    
                    if success:
                        relation_edges.append({
                            "source": source_text,
                            "source_type": source_type,
                            "target": target_text,
                            "target_type": target_type,
                            "type": relation_type,
                            "description": relation["description"]
                        })
                
                # 创建其他类型的关系
                # TODO: 实现其他类型关系的创建
                # 例如：合约的保证金比例、合约的交割日期、规则的生效日期等
                
            except Exception as e:
                logger.error(f"创建关系失败: {relation['type']} - {str(e)}")
        
        return relation_edges
    
    def _format_cypher_value(self, value):
        """格式化Cypher值，处理不同类型"""
        if value is None:
            return "NULL"
        elif isinstance(value, str):
            # 转义引号
            escaped = value.replace("'", "\\'")
            return f"'{escaped}'"
        elif isinstance(value, bool):
            return str(value).lower()
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, dict):
            # 将字典转换为Cypher映射
            props = ", ".join([f"{k}: {self._format_cypher_value(v)}" for k, v in value.items()])
            return f"{{{props}}}"
        elif isinstance(value, list):
            # 将列表转换为Cypher列表
            items = ", ".join([self._format_cypher_value(item) for item in value])
            return f"[{items}]"
        else:
            # 其他类型转为字符串
            return f"'{str(value)}'"
    
    def close(self):
        """关闭资源和连接"""
        if hasattr(self, 'knowledge_graph'):
            self.knowledge_graph.close()
        logger.info("知识提取与图谱构建流水线已关闭")


# 示例使用
if __name__ == "__main__":
    # 示例文本
    test_text = """
    根据规则RULE_2024_001，从2024年1月1日起，IF2406合约的保证金比例调整为8%，交割日为2024年6月15日。
    沪铜期货持仓限额为10000手。
    """
    
    # 创建知识提取流水线（使用测试连接）
    pipeline = KnowledgeExtractionPipeline(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        output_dir="./output"
    )
    
    # 处理文本
    metadata = {
        "file_type": "text",
        "file_name": "example.txt",
        "source": "用户输入"
    }
    
    result = pipeline.process_text(test_text, metadata)
    
    # 打印结果
    print(f"文档ID: {result.get('doc_id')}")
    print(f"提取到 {len(result.get('entities', []))} 个实体")
    print(f"提取到 {len(result.get('relations', []))} 条关系")
    
    # 关闭资源
    pipeline.close()