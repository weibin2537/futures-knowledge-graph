#!/usr/bin/env python3
"""
期货业务知识提取批处理脚本
用于从目录或文件列表中提取期货业务知识并构建知识图谱
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime
from pathlib import Path

# 导入自定义模块
from config_module import create_config
from knowledge_extraction_pipeline import KnowledgeExtractionPipeline

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_argparser():
    """设置命令行参数解析器"""
    parser = argparse.ArgumentParser(description='期货业务知识提取批处理工具')
    
    # 输入源参数
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--dir', '-d', help='要处理的目录路径')
    input_group.add_argument('--files', '-f', nargs='+', help='要处理的文件列表')
    input_group.add_argument('--text', '-t', help='要处理的文本内容')
    
    # 配置参数
    parser.add_argument('--config', '-c', help='配置文件路径')
    parser.add_argument('--env', '-e', help='环境名称（如dev, prod）')
    
    # 处理选项
    parser.add_argument('--recursive', '-r', action='store_true', help='是否递归处理子目录')
    parser.add_argument('--extensions', nargs='+', help='要处理的文件扩展名列表，如 .pdf .docx')
    parser.add_argument('--output', '-o', help='结果输出目录')
    parser.add_argument('--export', help='导出图谱数据的文件路径')
    parser.add_argument('--export-format', choices=['json', 'csv', 'cypher'], default='json', help='导出格式')
    
    # 其他选项
    parser.add_argument('--verbose', '-v', action='store_true', help='显示详细日志')
    
    return parser

def main():
    """主函数"""
    # 解析命令行参数
    parser = setup_argparser()
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 加载配置
    config = create_config(args.config, args.env)
    
    # 获取数据库配置
    db_config = config.get_database_config()
    neo4j_uri = db_config.get('neo4j_uri')
    neo4j_user = db_config.get('neo4j_user')
    neo4j_password = db_config.get('neo4j_password')
    
    # 获取模型配置
    models_config = config.get_models_config()
    entity_model_path = models_config.get('entity_model_path')
    relation_model_path = models_config.get('relation_model_path')
    
    # 设置输出目录
    output_dir = args.output or config.get('paths.output_dir')
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建知识提取流水线
    pipeline = KnowledgeExtractionPipeline(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        entity_model_path=entity_model_path,
        relation_model_path=relation_model_path,
        output_dir=output_dir
    )
    
    try:
        # 根据输入类型处理
        result = None
        
        if args.dir:
            # 处理目录
            extensions = args.extensions or config.get('processing.supported_extensions')
            logger.info(f"处理目录: {args.dir}")
            logger.info(f"文件类型: {extensions}")
            logger.info(f"递归处理: {args.recursive}")
            
            result = pipeline.extract_knowledge_from_directory(
                directory=args.dir,
                recursive=args.recursive,
                file_extensions=extensions
            )
        
        elif args.files:
            # 处理文件列表
            logger.info(f"处理文件列表: {args.files}")
            
            # 检查文件是否存在
            valid_files = []
            for file_path in args.files:
                if os.path.exists(file_path) and os.path.isfile(file_path):
                    valid_files.append(file_path)
                else:
                    logger.warning(f"文件不存在或不是文件: {file_path}")
            
            if not valid_files:
                logger.error("没有有效的文件可处理")
                return 1
            
            # 批量处理文件
            result = pipeline.batch_process(valid_files)
        
        elif args.text:
            # 处理文本
            logger.info(f"处理文本，长度: {len(args.text)}")
            
            # 设置元数据
            metadata = {
                "file_type": "text",
                "file_name": "command_line_input.txt",
                "source": "命令行输入",
                "timestamp": datetime.now().isoformat()
            }
            
            # 处理文本
            result = pipeline.process_text(args.text, metadata)
        
        # 显示结果摘要
        if result:
            if "error" in result:
                logger.error(f"处理失败: {result['error']}")
                return 1
            
            if args.dir or args.files:
                # 显示批处理结果
                logger.info(f"处理完成。总数: {result.get('total', 0)}, 成功: {result.get('success', 0)}, 失败: {result.get('failure', 0)}")
            else:
                # 显示单一文本处理结果
                logger.info(f"文档ID: {result.get('doc_id')}")
                logger.info(f"提取到 {len(result.get('entities', []))} 个实体")
                logger.info(f"提取到 {len(result.get('relations', []))} 条关系")
        
        # 如果指定了导出选项，导出图谱数据
        if args.export:
            logger.info(f"导出图谱数据到: {args.export}")
            export_result = pipeline.export_knowledge_graph(
                output_format=args.export_format,
                output_file=args.export
            )
            
            if "error" in export_result:
                logger.error(f"导出失败: {export_result['error']}")
            else:
                logger.info(f"导出成功: {export_result}")
    
    except Exception as e:
        logger.exception(f"处理过程中发生错误: {str(e)}")
        return 1
    
    finally:
        # 关闭资源
        pipeline.close()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())