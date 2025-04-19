from neo4j import GraphDatabase
import logging
import uuid
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KnowledgeGraph:
    """期货业务知识图谱管理类"""
    
    def __init__(self, uri, user, password):
        """
        初始化知识图谱连接
        
        Args:
            uri: Neo4j数据库URI
            user: 用户名
            password: 密码
        """
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            # 测试连接
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS test")
                for record in result:
                    if record["test"] == 1:
                        logger.info("Neo4j连接成功")
            
            # 初始化图谱模式
            self._init_schema()
        except Exception as e:
            logger.error(f"Neo4j连接失败: {str(e)}")
            raise ConnectionError(f"无法连接到Neo4j: {str(e)}")
    
    def close(self):
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j连接已关闭")
    
    def _init_schema(self):
        """初始化图谱模式，创建索引和约束"""
        with self.driver.session() as session:
            # 创建唯一性约束
            constraints = [
                # 合约代码唯一性约束
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:CONTRACT) REQUIRE c.contract_id IS UNIQUE",
                # 规则ID唯一性约束
                "CREATE CONSTRAINT IF NOT EXISTS FOR (r:RULE) REQUIRE r.rule_id IS UNIQUE",
                # 术语名称唯一性约束
                "CREATE CONSTRAINT IF NOT EXISTS FOR (t:TERM) REQUIRE t.term_name IS UNIQUE",
                # 品种名称唯一性约束
                "CREATE CONSTRAINT IF NOT EXISTS FOR (v:VARIETY) REQUIRE v.variety_name IS UNIQUE",
                # 事件ID唯一性约束
                "CREATE CONSTRAINT IF NOT EXISTS FOR (e:EVENT) REQUIRE e.event_id IS UNIQUE",
                # 文档ID唯一性约束
                "CREATE CONSTRAINT IF NOT EXISTS FOR (d:DOCUMENT) REQUIRE d.doc_id IS UNIQUE"
            ]
            
            # 创建索引
            indexes = [
                # 合约交割日索引
                "CREATE INDEX IF NOT EXISTS FOR (c:CONTRACT) ON (c.delivery_date)",
                # 规则生效日期索引
                "CREATE INDEX IF NOT EXISTS FOR (r:RULE) ON (r.effective_date)",
                # 事件发生日期索引
                "CREATE INDEX IF NOT EXISTS FOR (e:EVENT) ON (e.event_date)"
            ]
            
            # 执行约束和索引创建
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    logger.warning(f"创建约束失败: {str(e)}")
            
            for index in indexes:
                try:
                    session.run(index)
                except Exception as e:
                    logger.warning(f"创建索引失败: {str(e)}")
            
            logger.info("Neo4j图谱模式初始化完成")
    
    def create_document_node(self, doc_meta):
        """
        创建文档节点
        
        Args:
            doc_meta: 文档元数据字典
            
        Returns:
            str: 文档节点ID
        """
        with self.driver.session() as session:
            doc_id = str(uuid.uuid4())
            file_type = doc_meta.get("file_type", "unknown")
            file_name = doc_meta.get("source_file", "unknown")
            
            # 创建文档节点
            query = """
            CREATE (d:DOCUMENT {
                doc_id: $doc_id,
                file_type: $file_type,
                file_name: $file_name,
                import_time: $import_time,
                page_count: $page_count,
                metadata: $metadata
            })
            RETURN d.doc_id AS doc_id
            """
            
            # 准备参数
            params = {
                "doc_id": doc_id,
                "file_type": file_type,
                "file_name": file_name,
                "import_time": datetime.now().isoformat(),
                "page_count": doc_meta.get("page_count", 0),
                "metadata": {k: v for k, v in doc_meta.items() if k not in ["file_type", "source_file", "page_count"]}
            }
            
            # 执行查询
            result = session.run(query, params)
            record = result.single()
            if record:
                logger.info(f"已创建文档节点: {doc_id}")
                return record["doc_id"]
            else:
                logger.error("创建文档节点失败")
                return None
    
    def create_contract_node(self, contract_data, doc_id=None):
        """
        创建合约节点
        
        Args:
            contract_data: 合约数据字典
            doc_id: 相关文档ID，可选
            
        Returns:
            str: 合约节点ID
        """
        with self.driver.session() as session:
            contract_id = contract_data.get("contract_id")
            if not contract_id:
                logger.error("合约数据缺少contract_id字段")
                return None
            
            # 合约属性
            variety = contract_data.get("variety", "")
            delivery_date = contract_data.get("delivery_date", "")
            
            # 创建合约节点
            query = """
            MERGE (c:CONTRACT {contract_id: $contract_id})
            ON CREATE SET 
                c.variety = $variety,
                c.delivery_date = $delivery_date,
                c.created_at = $created_at
            ON MATCH SET 
                c.variety = CASE WHEN $variety <> "" THEN $variety ELSE c.variety END,
                c.delivery_date = CASE WHEN $delivery_date <> "" THEN $delivery_date ELSE c.delivery_date END,
                c.updated_at = $updated_at
            """
            
            params = {
                "contract_id": contract_id,
                "variety": variety,
                "delivery_date": delivery_date,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            # 执行合约节点创建查询
            session.run(query, params)
            
            # 如果提供了文档ID，创建合约与文档的关系
            if doc_id:
                rel_query = """
                MATCH (c:CONTRACT {contract_id: $contract_id})
                MATCH (d:DOCUMENT {doc_id: $doc_id})
                MERGE (c)-[r:MENTIONED_IN]->(d)
                ON CREATE SET r.created_at = $created_at
                """
                
                rel_params = {
                    "contract_id": contract_id,
                    "doc_id": doc_id,
                    "created_at": datetime.now().isoformat()
                }
                
                session.run(rel_query, rel_params)
            
            logger.info(f"已创建/更新合约节点: {contract_id}")
            return contract_id
    
    def create_rule_node(self, rule_data, doc_id=None):
        """
        创建规则节点
        
        Args:
            rule_data: 规则数据字典
            doc_id: 相关文档ID，可选
            
        Returns:
            str: 规则节点ID
        """
        with self.driver.session() as session:
            rule_id = rule_data.get("rule_id")
            if not rule_id:
                # 如果没有提供规则ID，生成一个
                current_year = datetime.now().year
                rule_id = f"RULE_{current_year}_{uuid.uuid4().hex[:4]}"
            
            # 规则属性
            effective_date = rule_data.get("effective_date", "")
            margin_ratio = rule_data.get("margin_ratio", 0.0)
            description = rule_data.get("description", "")
            
            # 创建规则节点
            query = """
            MERGE (r:RULE {rule_id: $rule_id})
            ON CREATE SET 
                r.effective_date = $effective_date,
                r.margin_ratio = $margin_ratio,
                r.description = $description,
                r.created_at = $created_at
            ON MATCH SET 
                r.effective_date = CASE WHEN $effective_date <> "" THEN $effective_date ELSE r.effective_date END,
                r.margin_ratio = CASE WHEN $margin_ratio <> 0.0 THEN $margin_ratio ELSE r.margin_ratio END,
                r.description = CASE WHEN $description <> "" THEN $description ELSE r.description END,
                r.updated_at = $updated_at
            RETURN r.rule_id AS rule_id
            """
            
            params = {
                "rule_id": rule_id,
                "effective_date": effective_date,
                "margin_ratio": margin_ratio,
                "description": description,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            # 执行规则节点创建查询
            result = session.run(query, params)
            record = result.single()
            
            # 如果提供了文档ID，创建规则与文档的关系
            if doc_id and record:
                rel_query = """
                MATCH (r:RULE {rule_id: $rule_id})
                MATCH (d:DOCUMENT {doc_id: $doc_id})
                MERGE (r)-[rel:DEFINED_IN]->(d)
                ON CREATE SET rel.created_at = $created_at
                """
                
                rel_params = {
                    "rule_id": rule_id,
                    "doc_id": doc_id,
                    "created_at": datetime.now().isoformat()
                }
                
                session.run(rel_query, rel_params)
            
            if record:
                logger.info(f"已创建/更新规则节点: {rule_id}")
                return record["rule_id"]
            else:
                logger.error("创建规则节点失败")
                return None
    
    def create_variety_node(self, variety_name, doc_id=None):
        """
        创建品种节点
        
        Args:
            variety_name: 品种名称
            doc_id: 相关文档ID，可选
            
        Returns:
            str: 品种节点ID
        """
        with self.driver.session() as session:
            # 创建品种节点
            query = """
            MERGE (v:VARIETY {variety_name: $variety_name})
            ON CREATE SET v.created_at = $created_at
            ON MATCH SET v.updated_at = $updated_at
            RETURN v.variety_name AS variety_name
            """
            
            params = {
                "variety_name": variety_name,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            # 执行品种节点创建查询
            result = session.run(query, params)
            record = result.single()
            
            # 如果提供了文档ID，创建品种与文档的关系
            if doc_id and record:
                rel_query = """
                MATCH (v:VARIETY {variety_name: $variety_name})
                MATCH (d:DOCUMENT {doc_id: $doc_id})
                MERGE (v)-[r:MENTIONED_IN]->(d)
                ON CREATE SET r.created_at = $created_at
                """
                
                rel_params = {
                    "variety_name": variety_name,
                    "doc_id": doc_id,
                    "created_at": datetime.now().isoformat()
                }
                
                session.run(rel_query, rel_params)
            
            if record:
                logger.info(f"已创建/更新品种节点: {variety_name}")
                return record["variety_name"]
            else:
                logger.error("创建品种节点失败")
                return None
    
    def create_term_node(self, term_data, doc_id=None):
        """
        创建术语节点
        
        Args:
            term_data: 术语数据字典
            doc_id: 相关文档ID，可选
            
        Returns:
            str: 术语节点ID
        """
        with self.driver.session() as session:
            term_name = term_data.get("term_name")
            if not term_name:
                logger.error("术语数据缺少term_name字段")
                return None
            
            # 术语属性
            definition = term_data.get("definition", "")
            
            # 创建术语节点
            query = """
            MERGE (t:TERM {term_name: $term_name})
            ON CREATE SET 
                t.definition = $definition,
                t.created_at = $created_at
            ON MATCH SET 
                t.definition = CASE WHEN $definition <> "" THEN $definition ELSE t.definition END,
                t.updated_at = $updated_at
            RETURN t.term_name AS term_name
            """
            
            params = {
                "term_name": term_name,
                "definition": definition,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            # 执行术语节点创建查询
            result = session.run(query, params)
            record = result.single()
            
            # 如果提供了文档ID，创建术语与文档的关系
            if doc_id and record:
                rel_query = """
                MATCH (t:TERM {term_name: $term_name})
                MATCH (d:DOCUMENT {doc_id: $doc_id})
                MERGE (t)-[r:EXPLAINED_IN]->(d)
                ON CREATE SET r.created_at = $created_at
                """
                
                rel_params = {
                    "term_name": term_name,
                    "doc_id": doc_id,
                    "created_at": datetime.now().isoformat()
                }
                
                session.run(rel_query, rel_params)
            
            if record:
                logger.info(f"已创建/更新术语节点: {term_name}")
                return record["term_name"]
            else:
                logger.error("创建术语节点失败")
                return None
    
    def create_event_node(self, event_data, doc_id=None):
        """
        创建事件节点
        
        Args:
            event_data: 事件数据字典
            doc_id: 相关文档ID，可选
            
        Returns:
            str: 事件节点ID
        """
        with self.driver.session() as session:
            event_name = event_data.get("event_name")
            if not event_name:
                logger.error("事件数据缺少event_name字段")
                return None
            
            # 事件属性
            event_date = event_data.get("event_date", "")
            event_id = f"EVENT_{uuid.uuid4().hex[:8]}"
            
            # 创建事件节点
            query = """
            CREATE (e:EVENT {
                event_id: $event_id,
                event_name: $event_name,
                event_date: $event_date,
                created_at: $created_at
            })
            RETURN e.event_id AS event_id
            """
            
            params = {
                "event_id": event_id,
                "event_name": event_name,
                "event_date": event_date,
                "created_at": datetime.now().isoformat()
            }
            
            # 执行事件节点创建查询
            result = session.run(query, params)
            record = result.single()
            
            # 如果提供了文档ID，创建事件与文档的关系
            if doc_id and record:
                rel_query = """
                MATCH (e:EVENT {event_id: $event_id})
                MATCH (d:DOCUMENT {doc_id: $doc_id})
                MERGE (e)-[r:MENTIONED_IN]->(d)
                ON CREATE SET r.created_at = $created_at
                """
                
                rel_params = {
                    "event_id": event_id,
                    "doc_id": doc_id,
                    "created_at": datetime.now().isoformat()
                }
                
                session.run(rel_query, rel_params)
            
            if record:
                logger.info(f"已创建事件节点: {event_id}")
                return record["event_id"]
            else:
                logger.error("创建事件节点失败")
                return None
    
    def create_relation(self, start_node, relation_type, end_node, properties=None):
        """
        创建两个节点之间的关系
        
        Args:
            start_node: 起始节点字典，包含label和key属性
            relation_type: 关系类型
            end_node: 结束节点字典，包含label和key属性
            properties: 关系属性，可选
            
        Returns:
            bool: 是否成功创建关系
        """
        with self.driver.session() as session:
            # 验证参数
            if not start_node or not relation_type or not end_node:
                logger.error("创建关系所需参数不完整")
                return False
            
            if not start_node.get("label") or not start_node.get("key") or not start_node.get("value"):
                logger.error("起始节点参数不完整")
                return False
            
            if not end_node.get("label") or not end_node.get("key") or not end_node.get("value"):
                logger.error("结束节点参数不完整")
                return False
            
            # 准备关系属性
            if properties is None:
                properties = {}
            
            properties["created_at"] = datetime.now().isoformat()
            
            # 构建Cypher参数
            props_str = ", ".join([f"r.{k} = ${k}" for k in properties.keys()])
            
            # 构建Cypher查询
            query = f"""
            MATCH (a:{start_node['label']} {{{start_node['key']}: $start_value}})
            MATCH (b:{end_node['label']} {{{end_node['key']}: $end_value}})
            MERGE (a)-[r:{relation_type}]->(b)
            ON CREATE SET {props_str}
            RETURN r
            """
            
            # 准备参数
            params = {
                "start_value": start_node["value"],
                "end_value": end_node["value"],
                **properties
            }
            
            # 执行查询
            try:
                result = session.run(query, params)
                return result.single() is not None
            except Exception as e:
                logger.error(f"创建关系失败: {str(e)}")
                return False
    
    def create_apply_relation(self, rule_id, contract_id, properties=None):
        """
        创建"规则适用于合约"的关系
        
        Args:
            rule_id: 规则ID
            contract_id: 合约ID
            properties: 关系属性，可选
            
        Returns:
            bool: 是否成功创建关系
        """
        start_node = {
            "label": "RULE",
            "key": "rule_id",
            "value": rule_id
        }
        
        end_node = {
            "label": "CONTRACT",
            "key": "contract_id",
            "value": contract_id
        }
        
        return self.create_relation(start_node, "APPLY_TO", end_node, properties)
    
    def search_contracts(self, query_params=None, limit=10):
        """
        搜索合约
        
        Args:
            query_params: 查询参数字典，可选
            limit: 结果限制，默认10
            
        Returns:
            list: 合约节点列表
        """
        with self.driver.session() as session:
            # 构建查询条件
            conditions = []
            params = {}
            
            if query_params:
                if "contract_id" in query_params:
                    conditions.append("c.contract_id CONTAINS $contract_id")
                    params["contract_id"] = query_params["contract_id"]
                
                if "variety" in query_params:
                    conditions.append("c.variety CONTAINS $variety")
                    params["variety"] = query_params["variety"]
                
                if "delivery_date" in query_params:
                    conditions.append("c.delivery_date = $delivery_date")
                    params["delivery_date"] = query_params["delivery_date"]
            
            # 构建Cypher查询
            where_clause = " AND ".join(conditions) if conditions else ""
            if where_clause:
                where_clause = f"WHERE {where_clause}"
            
            query = f"""
            MATCH (c:CONTRACT)
            {where_clause}
            RETURN c
            LIMIT $limit
            """
            
            params["limit"] = limit
            
            # 执行查询
            result = session.run(query, params)
            
            # 处理结果
            contracts = []
            for record in result:
                contract_node = record["c"]
                contract = dict(contract_node.items())
                contracts.append(contract)
            
            return contracts
    
    def search_rules(self, query_params=None, limit=10):
        """
        搜索规则
        
        Args:
            query_params: 查询参数字典，可选
            limit: 结果限制，默认10
            
        Returns:
            list: 规则节点列表
        """
        with self.driver.session() as session:
            # 构建查询条件
            conditions = []
            params = {}
            
            if query_params:
                if "rule_id" in query_params:
                    conditions.append("r.rule_id CONTAINS $rule_id")
                    params["rule_id"] = query_params["rule_id"]
                
                if "effective_date" in query_params:
                    conditions.append("r.effective_date = $effective_date")
                    params["effective_date"] = query_params["effective_date"]
                
                if "description" in query_params:
                    conditions.append("r.description CONTAINS $description")
                    params["description"] = query_params["description"]
            
            # 构建Cypher查询
            where_clause = " AND ".join(conditions) if conditions else ""
            if where_clause:
                where_clause = f"WHERE {where_clause}"
            
            query = f"""
            MATCH (r:RULE)
            {where_clause}
            RETURN r
            LIMIT $limit
            """
            
            params["limit"] = limit
            
            # 执行查询
            result = session.run(query, params)
            
            # 处理结果
            rules = []
            for record in result:
                rule_node = record["r"]
                rule = dict(rule_node.items())
                rules.append(rule)
            
            return rules
    
    def find_rules_for_contract(self, contract_id):
        """
        查找适用于特定合约的所有规则
        
        Args:
            contract_id: 合约ID
            
        Returns:
            list: 规则列表
        """
        with self.driver.session() as session:
            query = """
            MATCH (r:RULE)-[:APPLY_TO]->(c:CONTRACT {contract_id: $contract_id})
            RETURN r
            ORDER BY r.effective_date DESC
            """
            
            params = {
                "contract_id": contract_id
            }
            
            # 执行查询
            result = session.run(query, params)
            
            # 处理结果
            rules = []
            for record in result:
                rule_node = record["r"]
                rule = dict(rule_node.items())
                rules.append(rule)
            
            return rules
    
    def find_contracts_for_rule(self, rule_id):
        """
        查找特定规则适用的所有合约
        
        Args:
            rule_id: 规则ID
            
        Returns:
            list: 合约列表
        """
        with self.driver.session() as session:
            query = """
            MATCH (r:RULE {rule_id: $rule_id})-[:APPLY_TO]->(c:CONTRACT)
            RETURN c
            """
            
            params = {
                "rule_id": rule_id
            }
            
            # 执行查询
            result = session.run(query, params)
            
            # 处理结果
            contracts = []
            for record in result:
                contract_node = record["c"]
                contract = dict(contract_node.items())
                contracts.append(contract)
            
            return contracts
    
    def get_rule_contract_relations(self, effective_year=None):
        """
        获取规则-合约关系数据，用于前端可视化
        
        Args:
            effective_year: 规则生效年份，可选
            
        Returns:
            dict: 包含节点和边的数据
        """
        with self.driver.session() as session:
            # 构建查询条件
            where_clause = ""
            params = {}
            
            if effective_year:
                where_clause = "WHERE r.effective_date STARTS WITH $year_prefix"
                params["year_prefix"] = f"{effective_year}-"
            
            # 查询规则-合约关系
            query = f"""
            MATCH (r:RULE)-[rel:APPLY_TO]->(c:CONTRACT)
            {where_clause}
            RETURN r.rule_id AS rule_id, r.description AS rule_desc, 
                   c.contract_id AS contract_id, c.variety AS variety,
                   r.margin_ratio AS margin_ratio
            """
            
            # 执行查询
            result = session.run(query, params)
            
            # 处理结果
            nodes = []
            edges = []
            node_ids = set()
            
            for record in result:
                rule_id = record["rule_id"]
                contract_id = record["contract_id"]
                
                # 添加规则节点
                if rule_id not in node_ids:
                    nodes.append({
                        "id": rule_id,
                        "label": rule_id,
                        "title": record["rule_desc"] or rule_id,
                        "type": "rule"
                    })
                    node_ids.add(rule_id)
                
                # 添加合约节点
                if contract_id not in node_ids:
                    nodes.append({
                        "id": contract_id,
                        "label": contract_id,
                        "title": record["variety"] or contract_id,
                        "type": "contract"
                    })
                    node_ids.add(contract_id)
                
                # 添加边
                edges.append({
                    "source": rule_id,
                    "target": contract_id,
                    "label": f"{record['margin_ratio']:.0%}",
                    "value": record["margin_ratio"]
                })
            
            return {
                "nodes": nodes,
                "edges": edges
            }

    def execute_cypher(self, query, params=None):
        """
        执行自定义Cypher查询
        
        Args:
            query: Cypher查询字符串
            params: 查询参数，可选
            
        Returns:
            list: 查询结果列表
        """
        with self.driver.session() as session:
            try:
                result = session.run(query, params or {})
                
                # 将结果转换为列表
                records = []
                for record in result:
                    record_dict = {}
                    for key, value in record.items():
                        # 处理节点类型
                        if hasattr(value, "items"):
                            record_dict[key] = dict(value.items())
                        # 处理关系类型
                        elif hasattr(value, "type") and hasattr(value, "start_node"):
                            record_dict[key] = {
                                "type": value.type,
                                "properties": dict(value.items())
                            }
                        # 处理其他类型
                        else:
                            record_dict[key] = value
                    records.append(record_dict)
                
                return records
            except Exception as e:
                logger.error(f"执行Cypher查询出错: {str(e)}")
                raise e

# 使用示例
if __name__ == "__main__":
    # 连接到Neo4j数据库
    neo4j_uri = "bolt://localhost:7687"
    neo4j_user = "neo4j"
    neo4j_password = "password"
    
    try:
        # 创建知识图谱管理器
        kg = KnowledgeGraph(neo4j_uri, neo4j_user, neo4j_password)
        
        # 创建文档节点
        doc_meta = {
            "file_type": "pdf",
            "source_file": "contract_example.pdf",
            "page_count": 10
        }
        doc_id = kg.create_document_node(doc_meta)
        
        # 创建合约节点
        contract_data = {
            "contract_id": "IF2406",
            "variety": "沪深300股指期货",
            "delivery_date": "2024-06-15"
        }
        contract_id = kg.create_contract_node(contract_data, doc_id)
        
        # 创建规则节点
        rule_data = {
            "rule_id": "RULE_2024_001",
            "effective_date": "2024-01-01",
            "margin_ratio": 0.08,
            "description": "沪深300股指期货保证金调整"
        }
        rule_id = kg.create_rule_node(rule_data, doc_id)
        
        # 创建规则-合约关系
        kg.create_apply_relation(rule_id, contract_id)
        
        # 查询示例：查找适用于IF2406的所有规则
        rules = kg.find_rules_for_contract("IF2406")
        print(f"适用于IF2406的规则: {rules}")
        
        # 关闭连接
        kg.close()
    except Exception as e:
        print(f"示例运行失败: {str(e)}")