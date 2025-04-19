// src/components/SearchPanel.vue
<template>
  <div class="search-panel">
    <div class="panel-header">
      <h2>知识图谱搜索</h2>
      <div class="search-tabs">
        <button 
          :class="['tab-btn', { active: activeTab === 'keywords' }]"
          @click="activeTab = 'keywords'"
        >
          关键词搜索
        </button>
        <button 
          :class="['tab-btn', { active: activeTab === 'advanced' }]"
          @click="activeTab = 'advanced'"
        >
          高级搜索
        </button>
        <button 
          :class="['tab-btn', { active: activeTab === 'cypher' }]"
          @click="activeTab = 'cypher'"
        >
          Cypher查询
        </button>
      </div>
    </div>

    <div class="search-content">
      <!-- 关键词搜索 -->
      <div v-if="activeTab === 'keywords'" class="tab-content">
        <div class="search-input-group">
          <input 
            type="text" 
            v-model="keywordSearch.query" 
            placeholder="输入关键词搜索合约、规则、品种等"
            @keyup.enter="searchByKeyword"
          />
          <button class="search-btn" @click="searchByKeyword">
            搜索
          </button>
        </div>
        <div class="entity-type-filter">
          <label>
            <input 
              type="radio" 
              v-model="keywordSearch.entityType" 
              value="all"
            /> 全部
          </label>
          <label>
            <input 
              type="radio" 
              v-model="keywordSearch.entityType" 
              value="CONTRACT"
            /> 合约
          </label>
          <label>
            <input 
              type="radio" 
              v-model="keywordSearch.entityType" 
              value="RULE"
            /> 规则
          </label>
          <label>
            <input 
              type="radio" 
              v-model="keywordSearch.entityType" 
              value="VARIETY"
            /> 期货品种
          </label>
          <label>
            <input 
              type="radio" 
              v-model="keywordSearch.entityType" 
              value="TERM"
            /> 术语
          </label>
        </div>
      </div>

      <!-- 高级搜索 -->
      <div v-if="activeTab === 'advanced'" class="tab-content">
        <div class="advanced-search-form">
          <div class="form-group">
            <label>搜索类型</label>
            <select v-model="advancedSearch.type">
              <option value="contract">合约查询</option>
              <option value="rule">规则查询</option>
              <option value="relation">关系查询</option>
            </select>
          </div>

          <!-- 合约查询表单 -->
          <div v-if="advancedSearch.type === 'contract'" class="search-form">
            <div class="form-group">
              <label>合约代码</label>
              <input type="text" v-model="advancedSearch.contract.id" placeholder="如IF2406" />
            </div>
            <div class="form-group">
              <label>品种</label>
              <input type="text" v-model="advancedSearch.contract.variety" placeholder="如沪深300股指期货" />
            </div>
            <div class="form-group">
              <label>交割日期</label>
              <input type="date" v-model="advancedSearch.contract.deliveryDate" />
            </div>
          </div>

          <!-- 规则查询表单 -->
          <div v-if="advancedSearch.type === 'rule'" class="search-form">
            <div class="form-group">
              <label>规则ID</label>
              <input type="text" v-model="advancedSearch.rule.id" placeholder="如RULE_2024_001" />
            </div>
            <div class="form-group">
              <label>生效日期</label>
              <input type="date" v-model="advancedSearch.rule.effectiveDate" />
            </div>
            <div class="form-group">
              <label>保证金比例</label>
              <div class="range-input">
                <input 
                  type="range" 
                  v-model="advancedSearch.rule.marginRatio" 
                  min="0" 
                  max="30" 
                  step="1"
                />
                <span>{{ advancedSearch.rule.marginRatio }}%</span>
              </div>
            </div>
            <div class="form-group">
              <label>描述关键词</label>
              <input type="text" v-model="advancedSearch.rule.description" placeholder="如调整、涨跌停" />
            </div>
          </div>

          <!-- 关系查询表单 -->
          <div v-if="advancedSearch.type === 'relation'" class="search-form">
            <div class="form-group">
              <label>关系类型</label>
              <select v-model="advancedSearch.relation.type">
                <option value="APPLY_TO">规则适用于合约</option>
                <option value="MENTIONED_IN">在文档中提及</option>
                <option value="EXPLAINED_IN">在文档中解释</option>
              </select>
            </div>
            <div class="form-group">
              <label>起始节点</label>
              <input type="text" v-model="advancedSearch.relation.source" placeholder="输入节点ID" />
            </div>
            <div class="form-group">
              <label>结束节点</label>
              <input type="text" v-model="advancedSearch.relation.target" placeholder="输入节点ID" />
            </div>
          </div>

          <button class="search-btn full-width" @click="searchAdvanced">
            执行高级搜索
          </button>
        </div>
      </div>

      <!-- Cypher查询 -->
      <div v-if="activeTab === 'cypher'" class="tab-content">
        <div class="cypher-editor">
          <label>Cypher查询语句</label>
          <textarea 
            v-model="cypherSearch.query" 
            placeholder="输入Cypher查询语句，如: MATCH (r:RULE)-[:APPLY_TO]->(c:CONTRACT) RETURN r, c LIMIT 10"
            rows="6"
          ></textarea>
          
          <div class="cypher-templates">
            <label>常用查询模板:</label>
            <select @change="selectCypherTemplate($event)">
              <option value="">-- 选择查询模板 --</option>
              <option value="contract-rules">查询特定合约关联的规则</option>
              <option value="rule-contracts">查询特定规则适用的合约</option>
              <option value="margin-ratio">按保证金比例查询规则</option>
              <option value="variety-contracts">查询特定品种的所有合约</option>
              <option value="document-entities">查询文档中提及的实体</option>
            </select>
          </div>

          <button class="search-btn full-width" @click="executeCypher">
            执行Cypher查询
          </button>
        </div>
      </div>
    </div>

    <!-- 搜索结果 -->
    <div v-if="loading" class="loading">
      <div class="spinner"></div>
      <span>查询中...</span>
    </div>

    <div v-if="error" class="error-message">
      {{ error }}
    </div>

    <div v-if="searchResults.length > 0" class="search-results">
      <h3>查询结果 ({{ searchResults.length }})</h3>
      
      <div class="result-cards">
        <div 
          v-for="(result, index) in searchResults" 
          :key="index"
          class="result-card"
          :class="{ 'highlight': result.highlight }"
        >
          <div class="card-header">
            <span class="entity-type-badge" :class="result.type?.toLowerCase()">
              {{ getEntityTypeLabel(result.type) }}
            </span>
            <h4>{{ result.label || result.id }}</h4>
          </div>
          
          <div class="card-content">
            <div v-if="result.type === 'CONTRACT'" class="property-list">
              <div class="property">
                <span class="property-label">合约代码:</span>
                <span class="property-value">{{ result.contract_id || result.id }}</span>
              </div>
              <div class="property">
                <span class="property-label">品种:</span>
                <span class="property-value">{{ result.variety || '未指定' }}</span>
              </div>
              <div class="property">
                <span class="property-label">交割日:</span>
                <span class="property-value">{{ result.delivery_date || '未指定' }}</span>
              </div>
            </div>

            <div v-else-if="result.type === 'RULE'" class="property-list">
              <div class="property">
                <span class="property-label">规则ID:</span>
                <span class="property-value">{{ result.rule_id || result.id }}</span>
              </div>
              <div class="property">
                <span class="property-label">生效日期:</span>
                <span class="property-value">{{ result.effective_date || '未指定' }}</span>
              </div>
              <div class="property">
                <span class="property-label">保证金比例:</span>
                <span class="property-value">{{ result.margin_ratio ? `${(result.margin_ratio * 100).toFixed(0)}%` : '未指定' }}</span>
              </div>
              <div class="property">
                <span class="property-label">描述:</span>
                <span class="property-value">{{ result.description || '未指定' }}</span>
              </div>
            </div>

            <div v-else-if="result.type === 'RELATION'" class="property-list">
              <div class="property">
                <span class="property-label">关系类型:</span>
                <span class="property-value">{{ result.relation_type }}</span>
              </div>
              <div class="property">
                <span class="property-label">源节点:</span>
                <span class="property-value">{{ result.source }}</span>
              </div>
              <div class="property">
                <span class="property-label">目标节点:</span>
                <span class="property-value">{{ result.target }}</span>
              </div>
            </div>

            <div v-else class="property-list">
              <div 
                v-for="(value, key) in getDisplayProperties(result)" 
                :key="key" 
                class="property"
              >
                <span class="property-label">{{ formatPropertyName(key) }}:</span>
                <span class="property-value">{{ formatPropertyValue(value) }}</span>
              </div>
            </div>
          </div>
          
          <div class="card-actions">
            <button class="action-btn" @click="viewRelations(result)">查看关系</button>
            <button class="action-btn" @click="viewDocuments(result)">相关文档</button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { defineComponent, ref, reactive } from 'vue';
import axios from 'axios';

export default defineComponent({
  name: 'SearchPanel',
  emits: ['search-result'],
  props: {
    apiBaseUrl: {
      type: String,
      default: 'http://localhost:8000'
    }
  },
  setup(props, { emit }) {
    const activeTab = ref('keywords');
    const loading = ref(false);
    const error = ref('');
    const searchResults = ref([]);

    // 关键词搜索状态
    const keywordSearch = reactive({
      query: '',
      entityType: 'all'
    });

    // 高级搜索状态
    const advancedSearch = reactive({
      type: 'contract',
      contract: {
        id: '',
        variety: '',
        deliveryDate: ''
      },
      rule: {
        id: '',
        effectiveDate: '',
        marginRatio: 8,
        description: ''
      },
      relation: {
        type: 'APPLY_TO',
        source: '',
        target: ''
      }
    });

    // Cypher查询状态
    const cypherSearch = reactive({
      query: '',
      params: {}
    });

    // 关键词搜索
    const searchByKeyword = async () => {
      if (!keywordSearch.query.trim()) {
        error.value = '请输入搜索关键词';
        return;
      }

      loading.value = true;
      error.value = '';
      searchResults.value = [];

      try {
        let cypherQuery = '';
        const params = {};

        // 根据实体类型构建不同的查询
        if (keywordSearch.entityType === 'all') {
          // 搜索所有类型的实体
          cypherQuery = `
            MATCH (n)
            WHERE n.contract_id CONTAINS $query 
               OR n.rule_id CONTAINS $query 
               OR n.variety_name CONTAINS $query 
               OR n.term_name CONTAINS $query
               OR n.description CONTAINS $query
            RETURN n
            LIMIT 20
          `;
        } else if (keywordSearch.entityType === 'CONTRACT') {
          // 只搜索合约
          cypherQuery = `
            MATCH (c:CONTRACT)
            WHERE c.contract_id CONTAINS $query 
               OR c.variety CONTAINS $query
            RETURN c AS n
            LIMIT 20
          `;
        } else if (keywordSearch.entityType === 'RULE') {
          // 只搜索规则
          cypherQuery = `
            MATCH (r:RULE)
            WHERE r.rule_id CONTAINS $query 
               OR r.description CONTAINS $query
            RETURN r AS n
            LIMIT 20
          `;
        } else if (keywordSearch.entityType === 'VARIETY') {
          // 只搜索品种
          cypherQuery = `
            MATCH (v:VARIETY)
            WHERE v.variety_name CONTAINS $query
            RETURN v AS n
            LIMIT 20
          `;
        } else if (keywordSearch.entityType === 'TERM') {
          // 只搜索术语
          cypherQuery = `
            MATCH (t:TERM)
            WHERE t.term_name CONTAINS $query 
               OR t.definition CONTAINS $query
            RETURN t AS n
            LIMIT 20
          `;
        }

        params.query = keywordSearch.query;
        
        // 执行查询
        const response = await axios.post(`${props.apiBaseUrl}/cypher`, {
          query: cypherQuery,
          params: params
        });

        if (response.data && response.data.length > 0) {
          // 处理结果
          searchResults.value = response.data.map(record => {
            const node = record.n;
            return {
              ...node,
              type: getNodeType(node),
              label: getNodeLabel(node),
              highlight: isHighlighted(node, keywordSearch.query)
            };
          });

          // 触发搜索结果事件
          emit('search-result', searchResults.value);
        } else {
          error.value = '未找到匹配的结果';
        }
      } catch (err) {
        console.error('关键词搜索失败:', err);
        error.value = err.response?.data?.detail || '搜索失败，请重试';
      } finally {
        loading.value = false;
      }
    };

    // 高级搜索
    const searchAdvanced = async () => {
      loading.value = true;
      error.value = '';
      searchResults.value = [];

      try {
        let cypherQuery = '';
        const params = {};

        // 根据搜索类型构建不同的查询
        if (advancedSearch.type === 'contract') {
          cypherQuery = `
            MATCH (c:CONTRACT)
            WHERE 1=1
          `;

          if (advancedSearch.contract.id) {
            cypherQuery += ` AND c.contract_id CONTAINS $contractId`;
            params.contractId = advancedSearch.contract.id;
          }

          if (advancedSearch.contract.variety) {
            cypherQuery += ` AND c.variety CONTAINS $variety`;
            params.variety = advancedSearch.contract.variety;
          }

          if (advancedSearch.contract.deliveryDate) {
            cypherQuery += ` AND c.delivery_date = $deliveryDate`;
            params.deliveryDate = advancedSearch.contract.deliveryDate;
          }

          cypherQuery += `
            RETURN c AS n
            LIMIT 20
          `;
        } else if (advancedSearch.type === 'rule') {
          cypherQuery = `
            MATCH (r:RULE)
            WHERE 1=1
          `;

          if (advancedSearch.rule.id) {
            cypherQuery += ` AND r.rule_id CONTAINS $ruleId`;
            params.ruleId = advancedSearch.rule.id;
          }

          if (advancedSearch.rule.effectiveDate) {
            cypherQuery += ` AND r.effective_date = $effectiveDate`;
            params.effectiveDate = advancedSearch.rule.effectiveDate;
          }

          if (advancedSearch.rule.marginRatio > 0) {
            cypherQuery += ` AND r.margin_ratio >= $marginRatio`;
            params.marginRatio = advancedSearch.rule.marginRatio / 100; // 转换为小数
          }

          if (advancedSearch.rule.description) {
            cypherQuery += ` AND r.description CONTAINS $description`;
            params.description = advancedSearch.rule.description;
          }

          cypherQuery += `
            RETURN r AS n
            LIMIT 20
          `;
        } else if (advancedSearch.type === 'relation') {
          // 关系查询
          cypherQuery = `
            MATCH (a)-[r:${advancedSearch.relation.type}]->(b)
            WHERE 1=1
          `;

          if (advancedSearch.relation.source) {
            cypherQuery += ` AND (a.contract_id = $source OR a.rule_id = $source OR a.term_name = $source OR a.variety_name = $source)`;
            params.source = advancedSearch.relation.source;
          }

          if (advancedSearch.relation.target) {
            cypherQuery += ` AND (b.contract_id = $target OR b.rule_id = $target OR b.term_name = $target OR b.variety_name = $target)`;
            params.target = advancedSearch.relation.target;
          }

          cypherQuery += `
            RETURN a AS source, r AS relation, b AS target
            LIMIT 20
          `;
        }

        // 执行查询
        const response = await axios.post(`${props.apiBaseUrl}/cypher`, {
          query: cypherQuery,
          params: params
        });

        if (response.data && response.data.length > 0) {
          // 处理结果
          if (advancedSearch.type === 'relation') {
            // 处理关系查询结果
            searchResults.value = response.data.map(record => {
              return {
                type: 'RELATION',
                relation_type: record.relation.type,
                source: getNodeLabel(record.source),
                target: getNodeLabel(record.target),
                properties: record.relation.properties || {}
              };
            });
          } else {
            // 处理节点查询结果
            searchResults.value = response.data.map(record => {
              const node = record.n;
              return {
                ...node,
                type: getNodeType(node),
                label: getNodeLabel(node)
              };
            });
          }

          // 触发搜索结果事件
          emit('search-result', searchResults.value);
        } else {
          error.value = '未找到匹配的结果';
        }
      } catch (err) {
        console.error('高级搜索失败:', err);
        error.value = err.response?.data?.detail || '搜索失败，请重试';
      } finally {
        loading.value = false;
      }
    };

    // 执行Cypher查询
    const executeCypher = async () => {
      if (!cypherSearch.query.trim()) {
        error.value = '请输入Cypher查询语句';
        return;
      }

      loading.value = true;
      error.value = '';
      searchResults.value = [];

      try {
        const response = await axios.post(`${props.apiBaseUrl}/cypher`, {
          query: cypherSearch.query,
          params: cypherSearch.params
        });

        if (response.data && response.data.length > 0) {
          // 处理结果
          searchResults.value = response.data.map(record => {
            // 尝试提取节点数据
            const keys = Object.keys(record);
            const result = {};

            for (const key of keys) {
              const value = record[key];
              
              // 如果是节点
              if (value && typeof value === 'object' && !Array.isArray(value)) {
                Object.assign(result, value);
                if (!result.type) {
                  result.type = getNodeType(value);
                }
                if (!result.label) {
                  result.label = getNodeLabel(value);
                }
              } else {
                result[key] = value;
              }
            }

            return result;
          });

          // 触发搜索结果事件
          emit('search-result', searchResults.value);
        } else {
          error.value = '查询未返回结果';
        }
      } catch (err) {
        console.error('Cypher查询失败:', err);
        error.value = err.response?.data?.detail || '查询失败，请重试';
      } finally {
        loading.value = false;
      }
    };

    // 选择Cypher查询模板
    const selectCypherTemplate = (event) => {
      const templateValue = event.target.value;
      event.target.value = ''; // 重置选择框

      if (!templateValue) return;

      switch (templateValue) {
        case 'contract-rules':
          cypherSearch.query = `
MATCH (c:CONTRACT {contract_id: "IF2406"})-[:APPLY_TO]-(r:RULE)
RETURN c, r
LIMIT 10
          `.trim();
          break;
        case 'rule-contracts':
          cypherSearch.query = `
MATCH (r:RULE {rule_id: "RULE_2024_001"})-[:APPLY_TO]->(c:CONTRACT)
RETURN r, c
LIMIT 10
          `.trim();
          break;
        case 'margin-ratio':
          cypherSearch.query = `
MATCH (r:RULE)
WHERE r.margin_ratio > 0.05
RETURN r
ORDER BY r.margin_ratio DESC
LIMIT 10
          `.trim();
          break;
        case 'variety-contracts':
          cypherSearch.query = `
MATCH (c:CONTRACT)
WHERE c.variety CONTAINS "沪深300"
RETURN c
ORDER BY c.delivery_date
LIMIT 10
          `.trim();
          break;
        case 'document-entities':
          cypherSearch.query = `
MATCH (d:DOCUMENT {file_name: "contract_example.pdf"})<-[:MENTIONED_IN]-(n)
RETURN d, n
LIMIT 20
          `.trim();
          break;
      }
    };

    // 查看实体相关关系
    const viewRelations = (entity) => {
      if (!entity || !entity.type) return;

      // 根据实体类型构建关系查询
      let cypherQuery = '';
      const params = {};

      if (entity.type === 'CONTRACT') {
        cypherQuery = `
          MATCH (c:CONTRACT {contract_id: $id})-[r]-(n)
          RETURN c, r, n
          LIMIT 10
        `;
        params.id = entity.contract_id || entity.id;
      } else if (entity.type === 'RULE') {
        cypherQuery = `
          MATCH (r:RULE {rule_id: $id})-[rel]-(n)
          RETURN r, rel, n
          LIMIT 10
        `;
        params.id = entity.rule_id || entity.id;
      } else if (entity.type === 'VARIETY') {
        cypherQuery = `
          MATCH (v:VARIETY {variety_name: $id})-[r]-(n)
          RETURN v, r, n
          LIMIT 10
        `;
        params.id = entity.variety_name || entity.id;
      } else if (entity.type === 'TERM') {
        cypherQuery = `
          MATCH (t:TERM {term_name: $id})-[r]-(n)
          RETURN t, r, n
          LIMIT 10
        `;
        params.id = entity.term_name || entity.id;
      }

      // 更新Cypher查询
      cypherSearch.query = cypherQuery;
      cypherSearch.params = params;
      
      // 切换到Cypher查询标签
      activeTab.value = 'cypher';
      
      // 执行查询
      executeCypher();
    };

    // 查看实体相关文档
    const viewDocuments = (entity) => {
      if (!entity || !entity.type) return;

      // 根据实体类型构建文档查询
      let cypherQuery = '';
      const params = {};

      if (entity.type === 'CONTRACT') {
        cypherQuery = `
          MATCH (c:CONTRACT {contract_id: $id})-[:MENTIONED_IN]->(d:DOCUMENT)
          RETURN c, d
          LIMIT 10
        `;
        params.id = entity.contract_id || entity.id;
      } else if (entity.type === 'RULE') {
        cypherQuery = `
          MATCH (r:RULE {rule_id: $id})-[:DEFINED_IN]->(d:DOCUMENT)
          RETURN r, d
          LIMIT 10
        `;
        params.id = entity.rule_id || entity.id;
      } else if (entity.type === 'VARIETY') {
        cypherQuery = `
          MATCH (v:VARIETY {variety_name: $id})-[:MENTIONED_IN]->(d:DOCUMENT)
          RETURN v, d
          LIMIT 10
        `;
        params.id = entity.variety_name || entity.id;
      } else if (entity.type === 'TERM') {
        cypherQuery = `
          MATCH (t:TERM {term_name: $id})-[:EXPLAINED_IN]->(d:DOCUMENT)
          RETURN t, d
          LIMIT 10
        `;
        params.id = entity.term_name || entity.id;
      }

      // 更新Cypher查询
      cypherSearch.query = cypherQuery;
      cypherSearch.params = params;
      
      // 切换到Cypher查询标签
      activeTab.value = 'cypher';
      
      // 执行查询
      executeCypher();
    };

    // 获取节点类型
    const getNodeType = (node) => {
      if (node.contract_id) return 'CONTRACT';
      if (node.rule_id) return 'RULE';
      if (node.variety_name) return 'VARIETY';
      if (node.term_name) return 'TERM';
      if (node.doc_id) return 'DOCUMENT';
      if (node.event_id) return 'EVENT';
      return 'UNKNOWN';
    };

    // 获取节点标签
    const getNodeLabel = (node) => {
      if (node.contract_id) return node.contract_id;
      if (node.rule_id) return node.rule_id;
      if (node.variety_name) return node.variety_name;
      if (node.term_name) return node.term_name;
      if (node.doc_id) return node.file_name || node.doc_id;
      if (node.event_id) return node.event_name || node.event_id;
      return 'Unknown';
    };

    // 检查节点是否应该高亮
    const isHighlighted = (node, query) => {
      if (!query) return false;
      
      // 检查各字段是否包含查询关键词
      for (const key in node) {
        if (typeof node[key] === 'string' && node[key].includes(query)) {
          return true;
        }
      }
      
      return false;
    };

    // 获取实体类型标签
    const getEntityTypeLabel = (type) => {
      const typeMap = {
        'CONTRACT': '合约',
        'RULE': '规则',
        'VARIETY': '品种',
        'TERM': '术语',
        'DOCUMENT': '文档',
        'EVENT': '事件',
        'RELATION': '关系',
        'UNKNOWN': '未知'
      };
      
      return typeMap[type] || type;
    };

    // 获取显示属性
    const getDisplayProperties = (result) => {
      // 过滤掉内部属性和已经显示的属性
      const excluded = ['id', 'type', 'label', 'highlight'];
      const displayProps = {};
      
      for (const key in result) {
        if (!excluded.includes(key) && !key.startsWith('_')) {
          displayProps[key] = result[key];
        }
      }
      
      return displayProps;
    };

    // 格式化属性名称
    const formatPropertyName = (name) => {
      // 将下划线替换为空格，首字母大写
      return name
        .replace(/_/g, ' ')
        .replace(/\b\w/g, c => c.toUpperCase());
    };

    // 格式化属性值
    const formatPropertyValue = (value) => {
      if (value === null || value === undefined) {
        return '未指定';
      }
      
      if (typeof value === 'boolean') {
        return value ? '是' : '否';
      }
      
      if (typeof value === 'number') {
        // 如果可能是保证金比例，格式化为百分比
        if (value < 1) {
          return `${(value * 100).toFixed(0)}%`;
        }
        return value.toString();
      }
      
      return value;
    };

    return {
      activeTab,
      loading,
      error,
      searchResults,
      keywordSearch,
      advancedSearch,
      cypherSearch,
      searchByKeyword,
      searchAdvanced,
      executeCypher,
      selectCypherTemplate,
      viewRelations,
      viewDocuments,
      getEntityTypeLabel,
      getDisplayProperties,
      formatPropertyName,
      formatPropertyValue
    };
  }
});
</script>