// src/components/KnowledgeGraph.vue
<template>
  <div class="knowledge-graph-container">
    <div class="graph-controls">
      <div class="filter-section">
        <h3>筛选选项</h3>
        <div class="filter-group">
          <label>保证金比例:</label>
          <div class="range-slider">
            <input 
              type="range" 
              v-model="minMarginFilter" 
              :min="0" 
              :max="30" 
              step="1"
              @input="updateFilters"
            />
            <span>{{ minMarginFilter }}%</span>
          </div>
        </div>
        <div class="filter-group">
          <label>节点类型:</label>
          <div class="checkbox-group">
            <label>
              <input type="checkbox" v-model="showContracts" @change="updateFilters" />
              合约
            </label>
            <label>
              <input type="checkbox" v-model="showRules" @change="updateFilters" />
              规则
            </label>
          </div>
        </div>
      </div>
      <div class="legend">
        <div class="legend-item">
          <span class="color-box contract"></span>
          <span>合约</span>
        </div>
        <div class="legend-item">
          <span class="color-box rule"></span>
          <span>规则</span>
        </div>
      </div>
    </div>
    
    <div class="graph-wrapper" ref="graphContainer">
      <div v-if="loading" class="loading-overlay">
        <div class="spinner"></div>
        <div>加载图谱数据中...</div>
      </div>
      <div v-if="error" class="error-message">{{ error }}</div>
      <div v-if="!loading && !error && (!filteredNodes.length || !filteredEdges.length)" class="no-data">
        <p>暂无符合条件的图谱数据</p>
      </div>
      <div id="graph-canvas" ref="graphCanvas"></div>
    </div>
    
    <div class="node-details" v-if="selectedNode">
      <h3>{{ selectedNode.type === 'contract' ? '合约详情' : '规则详情' }}</h3>
      <div class="detail-item">
        <strong>ID:</strong> {{ selectedNode.id }}
      </div>
      <div class="detail-item" v-if="selectedNode.type === 'contract'">
        <strong>品种:</strong> {{ selectedNode.title || '未知' }}
      </div>
      <div class="detail-item" v-if="selectedNode.type === 'rule'">
        <strong>描述:</strong> {{ selectedNode.title || '未知' }}
      </div>
      <div class="related-nodes" v-if="relatedNodes.length">
        <h4>{{ selectedNode.type === 'contract' ? '关联规则' : '适用合约' }}</h4>
        <ul>
          <li v-for="node in relatedNodes" :key="node.id" @click="selectNode(node)">
            {{ node.label }} {{ node.value ? `(${(node.value * 100).toFixed(0)}%)` : '' }}
          </li>
        </ul>
      </div>
      <button class="close-btn" @click="clearSelection">关闭</button>
    </div>
  </div>
</template>

<script>
import { defineComponent, ref, onMounted, watch, computed } from 'vue';
import * as echarts from 'echarts';

export default defineComponent({
  name: 'KnowledgeGraph',
  props: {
    graphData: {
      type: Object,
      default: () => ({ nodes: [], edges: [] })
    },
    loading: {
      type: Boolean,
      default: false
    },
    error: {
      type: String,
      default: ''
    }
  },
  setup(props) {
    const graphCanvas = ref(null);
    const graphContainer = ref(null);
    let graphChart = null;
    
    // 筛选条件
    const minMarginFilter = ref(0);
    const showContracts = ref(true);
    const showRules = ref(true);
    const selectedNode = ref(null);
    const relatedNodes = ref([]);
    
    // 计算筛选后的节点和边
    const filteredNodes = computed(() => {
      if (!props.graphData || !props.graphData.nodes) return [];
      
      return props.graphData.nodes.filter(node => {
        // 根据节点类型筛选
        if (node.type === 'contract' && !showContracts.value) return false;
        if (node.type === 'rule' && !showRules.value) return false;
        return true;
      });
    });
    
    const filteredEdges = computed(() => {
      if (!props.graphData || !props.graphData.edges) return [];
      
      // 获取筛选后节点的ID集合
      const nodeIds = new Set(filteredNodes.value.map(node => node.id));
      
      return props.graphData.edges.filter(edge => {
        // 只保留连接筛选后节点的边
        if (!nodeIds.has(edge.source) || !nodeIds.has(edge.target)) return false;
        
        // 根据保证金比例筛选
        if (edge.value && edge.value * 100 < minMarginFilter.value) return false;
        return true;
      });
    });
    
    // 重新渲染图表
    const renderGraph = () => {
      if (!graphChart || !filteredNodes.value.length) return;
      
      const option = {
        tooltip: {
          trigger: 'item',
          formatter: (params) => {
            if (params.dataType === 'node') {
              return `<div style="font-weight:bold">${params.data.label}</div>
                     <div>${params.data.title || ''}</div>`;
            } else {
              // 边的提示信息
              return `<div>保证金比例: ${params.data.label || '未知'}</div>`;
            }
          }
        },
        legend: {
          show: false
        },
        animationDuration: 1500,
        animationEasingUpdate: 'quinticInOut',
        series: [{
          type: 'graph',
          layout: 'force',
          data: filteredNodes.value.map(node => ({
            ...node,
            symbolSize: node.type === 'rule' ? 30 : 20,
            itemStyle: {
              color: node.type === 'rule' ? '#C23531' : '#2F4554'
            }
          })),
          links: filteredEdges.value.map(edge => ({
            ...edge,
            lineStyle: {
              width: edge.value ? Math.max(1, edge.value * 10) : 1,
              color: '#aaa'
            }
          })),
          categories: [
            { name: '规则' },
            { name: '合约' }
          ],
          roam: true,
          label: {
            show: true,
            position: 'right',
            formatter: '{b}'
          },
          force: {
            repulsion: 200,
            edgeLength: 120
          },
          emphasis: {
            focus: 'adjacency',
            lineStyle: {
              width: 5
            }
          }
        }]
      };
      
      graphChart.setOption(option);
      
      // 添加点击事件
      graphChart.on('click', (params) => {
        if (params.dataType === 'node') {
          selectNode(params.data);
        }
      });
    };
    
    // 选中节点
    const selectNode = (node) => {
      selectedNode.value = node;
      
      // 获取关联节点
      if (props.graphData && props.graphData.edges) {
        const nodeId = node.id;
        const related = [];
        
        props.graphData.edges.forEach(edge => {
          // 如果当前节点是源节点，找到目标节点
          if (edge.source === nodeId) {
            const targetNode = props.graphData.nodes.find(n => n.id === edge.target);
            if (targetNode) {
              related.push({
                ...targetNode,
                value: edge.value
              });
            }
          }
          // 如果当前节点是目标节点，找到源节点
          else if (edge.target === nodeId) {
            const sourceNode = props.graphData.nodes.find(n => n.id === edge.source);
            if (sourceNode) {
              related.push({
                ...sourceNode,
                value: edge.value
              });
            }
          }
        });
        
        relatedNodes.value = related;
      }
    };
    
    // 清除选中
    const clearSelection = () => {
      selectedNode.value = null;
      relatedNodes.value = [];
    };
    
    // 更新筛选
    const updateFilters = () => {
      renderGraph();
    };
    
    onMounted(() => {
      if (graphCanvas.value) {
        graphChart = echarts.init(graphCanvas.value);
        
        // 容器调整大小时重新渲染
        window.addEventListener('resize', () => {
          if (graphChart) {
            graphChart.resize();
          }
        });
        
        renderGraph();
      }
    });
    
    watch(() => props.graphData, () => {
      renderGraph();
    }, { deep: true });
    
    watch([() => filteredNodes.value, () => filteredEdges.value], () => {
      renderGraph();
    });
    
    return {
      graphCanvas,
      graphContainer,
      minMarginFilter,
      showContracts,
      showRules,
      selectedNode,
      relatedNodes,
      filteredNodes,
      filteredEdges,
      updateFilters,
      selectNode,
      clearSelection
    };
  }
});
</script>

<style scoped>
.knowledge-graph-container {
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 600px;
  border: 1px solid #eaeaea;
  border-radius: 8px;
  overflow: hidden;
  background-color: #f9f9f9;
}

.graph-controls {
  display: flex;
  justify-content: space-between;
  padding: 16px;
  border-bottom: 1px solid #eaeaea;
  background-color: #fff;
}

.filter-section {
  flex: 1;
}

.filter-section h3 {
  margin-top: 0;
  margin-bottom: 12px;
  font-size: 16px;
}

.filter-group {
  margin-bottom: 12px;
}

.range-slider {
  display: flex;
  align-items: center;
}

.range-slider input {
  margin-right: 8px;
}

.checkbox-group {
  display: flex;
  gap: 16px;
}

.legend {
  display: flex;
  align-items: center;
  gap: 12px;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 6px;
}

.color-box {
  width: 16px;
  height: 16px;
  border-radius: 4px;
}

.color-box.rule {
  background-color: #C23531;
}

.color-box.contract {
  background-color: #2F4554;
}

.graph-wrapper {
  position: relative;
  flex: 1;
  overflow: hidden;
}

#graph-canvas {
  width: 100%;
  height: 100%;
  min-height: 500px;
}

.loading-overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  background-color: rgba(255, 255, 255, 0.8);
  z-index: 10;
}

.spinner {
  width: 40px;
  height: 40px;
  margin-bottom: 16px;
  border: 4px solid #f3f3f3;
  border-top: 4px solid #3498db;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.error-message {
  padding: 16px;
  text-align: center;
  color: #e74c3c;
}

.no-data {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100%;
  color: #7f8c8d;
}

.node-details {
  position: absolute;
  top: 20px;
  right: 20px;
  width: 300px;
  padding: 16px;
  background-color: white;
  border-radius: 8px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  z-index: 5;
}

.node-details h3 {
  margin-top: 0;
  margin-bottom: 16px;
  border-bottom: 1px solid #eee;
  padding-bottom: 8px;
}

.detail-item {
  margin-bottom: 8px;
}

.related-nodes {
  margin-top: 16px;
}

.related-nodes h4 {
  margin-bottom: 8px;
}

.related-nodes ul {
  margin: 0;
  padding-left: 20px;
}

.related-nodes li {
  cursor: pointer;
  margin-bottom: 4px;
  color: #2980b9;
}

.related-nodes li:hover {
  text-decoration: underline;
}

.close-btn {
  display: block;
  margin-top: 16px;
  padding: 6px 12px;
  background-color: #f3f3f3;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.close-btn:hover {
  background-color: #e0e0e0;
}
</style>