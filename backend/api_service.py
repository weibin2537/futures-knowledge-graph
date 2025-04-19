from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os
import uuid
import logging
from datetime import datetime

# 导入自定义模块
from document_parser import ParserFactory
from entity_extraction import EntityExtractor
from knowledge_graph import KnowledgeGraph

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="期货业务知识库API",
    description="期货业务知识库项目的后端API服务",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应限制为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 临时文件存储路径
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Neo4j连接配置
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# 全局服务实例
entity_extractor = EntityExtractor()

# 依赖项：获取知识图谱连接
def get_knowledge_graph():
    kg = KnowledgeGraph(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        yield kg
    finally:
        kg.close()

# 数据模型
class DocumentMetadata(BaseModel):
    doc_id: str
    file_type: str
    file_name: str
    import_time: str
    page_count: Optional[int] = None


class Entity(BaseModel):
    type: str
    label: str
    text: str
    start: int
    end: int
    score: float


class ContractData(BaseModel):
    contract_id: str
    variety: Optional[str] = None
    delivery_date: Optional[str] = None


class RuleData(BaseModel):
    rule_id: Optional[str] = None
    effective_date: Optional[str] = None
    margin_ratio: Optional[float] = None
    description: Optional[str] = None


class GraphNode(BaseModel):
    id: str
    label: str
    title: Optional[str] = None
    type: str


class GraphEdge(BaseModel):
    source: str
    target: str
    label: Optional[str] = None
    value: Optional[float] = None


class GraphData(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]


class SearchParams(BaseModel):
    query: Optional[str] = None
    entity_type: Optional[str] = None
    limit: int = 10


class CypherQuery(BaseModel):
    query: str
    params: Optional[Dict[str, Any]] = None


@app.get("/")
async def root():
    """API根路径，返回基本信息"""
    return {
        "message": "期货业务知识库API服务正在运行",
        "version": "1.0.0",
        "documentation": "/docs",
    }


@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    kg: KnowledgeGraph = Depends(get_knowledge_graph)
):
    """
    上传并解析文档
    
    - **file**: 要上传的文档文件
    """
    try:
        # 生成唯一文件名
        file_ext = os.path.splitext(file.filename)[1]
        temp_filename = f"{uuid.uuid4().hex}{file_ext}"
        file_path = os.path.join(UPLOAD_DIR, temp_filename)
        
        # 保存上传文件
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # 创建对应的解析器
        try:
            parser = ParserFactory.create_parser(file_path)
        except ValueError as e:
            os.remove(file_path)
            raise HTTPException(status_code=400, detail=f"不支持的文件类型: {str(e)}")
        
        # 解析文档
        parse_result = parser.parse(file_path)
        
        # 如果解析出错
        if "error" in parse_result:
            os.remove(file_path)
            raise HTTPException(status_code=500, detail=f"文档解析失败: {parse_result['error']}")
        
        # 提取实体
        if "text" in parse_result:
            entities = entity_extractor.extract_entities(parse_result["text"])
            parse_result["entities"] = entities
        
        # 创建文档节点
        doc_id = kg.create_document_node(parse_result["meta"])
        
        # 处理实体并创建图谱节点
        process_entities_to_graph(kg, parse_result["entities"], doc_id)
        
        # 删除临时文件
        os.remove(file_path)
        
        return {
            "message": "文档上传并解析成功",
            "doc_id": doc_id,
            "metadata": parse_result["meta"],
            "entity_count": len(parse_result["entities"])
        }
    except Exception as e:
        logger.error(f"文档上传处理失败: {str(e)}")
        # 确保临时文件被删除
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")


@app.get("/entities/extract", response_model=List[Entity])
async def extract_entities(text: str = Query(..., min_length=1)):
    """
    从文本中提取实体
    
    - **text**: 要提取实体的文本
    """
    try:
        entities = entity_extractor.extract_entities(text)
        return entities
    except Exception as e:
        logger.error(f"实体提取失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"实体提取失败: {str(e)}")


@app.post("/contracts", response_model=str)
async def create_contract(
    contract: ContractData,
    doc_id: Optional[str] = None,
    kg: KnowledgeGraph = Depends(get_knowledge_graph)
):
    """
    创建合约节点
    
    - **contract**: 合约数据
    - **doc_id**: 关联的文档ID，可选
    """
    try:
        contract_id = kg.create_contract_node(contract.dict(), doc_id)
        return contract_id
    except Exception as e:
        logger.error(f"创建合约节点失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"创建合约节点失败: {str(e)}")


@app.post("/rules", response_model=str)
async def create_rule(
    rule: RuleData,
    doc_id: Optional[str] = None,
    kg: KnowledgeGraph = Depends(get_knowledge_graph)
):
    """
    创建规则节点
    
    - **rule**: 规则数据
    - **doc_id**: 关联的文档ID，可选
    """
    try:
        rule_id = kg.create_rule_node(rule.dict(), doc_id)
        return rule_id
    except Exception as e:
        logger.error(f"创建规则节点失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"创建规则节点失败: {str(e)}")


@app.post("/relations/apply")
async def create_apply_relation(
    rule_id: str = Query(...),
    contract_id: str = Query(...),
    kg: KnowledgeGraph = Depends(get_knowledge_graph)
):
    """
    创建规则适用于合约的关系
    
    - **rule_id**: 规则ID
    - **contract_id**: 合约ID
    """
    try:
        success = kg.create_apply_relation(rule_id, contract_id)
        if success:
            return {"message": "关系创建成功"}
        else:
            raise HTTPException(status_code=500, detail="关系创建失败")
    except Exception as e:
        logger.error(f"创建关系失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"创建关系失败: {str(e)}")


@app.get("/contracts/search", response_model=List[Dict[str, Any]])
async def search_contracts(
    contract_id: Optional[str] = None,
    variety: Optional[str] = None,
    delivery_date: Optional[str] = None,
    limit: int = Query(10, ge=1, le=100),
    kg: KnowledgeGraph = Depends(get_knowledge_graph)
):
    """
    搜索合约
    
    - **contract_id**: 合约ID关键词，可选
    - **variety**: 品种关键词，可选
    - **delivery_date**: 交割日期，可选
    - **limit**: 结果数量限制，默认10
    """
    try:
        query_params = {}
        if contract_id:
            query_params["contract_id"] = contract_id
        if variety:
            query_params["variety"] = variety
        if delivery_date:
            query_params["delivery_date"] = delivery_date
        
        contracts = kg.search_contracts(query_params, limit)
        return contracts
    except Exception as e:
        logger.error(f"搜索合约失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"搜索合约失败: {str(e)}")


@app.get("/rules/search", response_model=List[Dict[str, Any]])
async def search_rules(
    rule_id: Optional[str] = None,
    effective_date: Optional[str] = None,
    description: Optional[str] = None,
    limit: int = Query(10, ge=1, le=100),
    kg: KnowledgeGraph = Depends(get_knowledge_graph)
):
    """
    搜索规则
    
    - **rule_id**: 规则ID关键词，可选
    - **effective_date**: 生效日期，可选
    - **description**: 描述关键词，可选
    - **limit**: 结果数量限制，默认10
    """
    try:
        query_params = {}
        if rule_id:
            query_params["rule_id"] = rule_id
        if effective_date:
            query_params["effective_date"] = effective_date
        if description:
            query_params["description"] = description
        
        rules = kg.search_rules(query_params, limit)
        return rules
    except Exception as e:
        logger.error(f"搜索规则失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"搜索规则失败: {str(e)}")


@app.get("/contracts/{contract_id}/rules", response_model=List[Dict[str, Any]])
async def get_rules_for_contract(
    contract_id: str,
    kg: KnowledgeGraph = Depends(get_knowledge_graph)
):
    """
    获取适用于特定合约的所有规则
    
    - **contract_id**: 合约ID
    """
    try:
        rules = kg.find_rules_for_contract(contract_id)
        return rules
    except Exception as e:
        logger.error(f"查询合约规则失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"查询合约规则失败: {str(e)}")


@app.get("/rules/{rule_id}/contracts", response_model=List[Dict[str, Any]])
async def get_contracts_for_rule(
    rule_id: str,
    kg: KnowledgeGraph = Depends(get_knowledge_graph)
):
    """
    获取特定规则适用的所有合约
    
    - **rule_id**: 规则ID
    """
    try:
        contracts = kg.find_contracts_for_rule(rule_id)
        return contracts
    except Exception as e:
        logger.error(f"查询规则适用合约失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"查询规则适用合约失败: {str(e)}")


@app.get("/graph/rule-contract", response_model=GraphData)
async def get_rule_contract_graph(
    effective_year: Optional[int] = None,
    kg: KnowledgeGraph = Depends(get_knowledge_graph)
):
    """
    获取规则-合约关系图数据，用于前端可视化
    
    - **effective_year**: 规则生效年份，可选
    """
    try:
        graph_data = kg.get_rule_contract_relations(effective_year)
        return graph_data
    except Exception as e:
        logger.error(f"获取图谱数据失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取图谱数据失败: {str(e)}")


@app.post("/cypher", response_model=List[Dict[str, Any]])
async def execute_cypher_query(
    cypher: CypherQuery,
    kg: KnowledgeGraph = Depends(get_knowledge_graph)
):
    """
    执行自定义Cypher查询
    
    - **cypher**: Cypher查询及参数
    """
    try:
        results = kg.execute_cypher(cypher.query, cypher.params)
        return results
    except Exception as e:
        logger.error(f"执行Cypher查询失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"执行Cypher查询失败: {str(e)}")


# 辅助函数：处理实体并创建图谱节点
def process_entities_to_graph(kg, entities, doc_id):
    """
    处理提取的实体，创建相应的图谱节点
    
    Args:
        kg: 知识图谱管理器
        entities: 实体列表
        doc_id: 文档ID
    """
    for entity in entities:
        entity_type = entity["type"]
        entity_text = entity["text"]
        
        # 根据实体类型创建不同类型的节点
        if entity_type == "CONTRACT":
            # 创建合约节点
            contract_data = {
                "contract_id": entity_text,
                "variety": "",  # 需要从上下文中提取或关联
                "delivery_date": ""  # 需要从上下文中提取或关联
            }
            kg.create_contract_node(contract_data, doc_id)
        
        elif entity_type == "RULE":
            # 创建规则节点
            rule_data = {
                "rule_id": entity_text,
                "effective_date": "",  # 需要从上下文中提取或关联
                "description": ""  # 需要从上下文中提取或关联
            }
            kg.create_rule_node(rule_data, doc_id)
        
        elif entity_type == "VARIETY":
            # 创建品种节点
            kg.create_variety_node(entity_text, doc_id)
        
        elif entity_type == "TERM":
            # 创建术语节点
            term_data = {
                "term_name": entity_text,
                "definition": ""  # 需要从上下文中提取或关联
            }
            kg.create_term_node(term_data, doc_id)


# 启动服务器
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)