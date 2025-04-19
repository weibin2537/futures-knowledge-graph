// src/App.vue
<template>
  <div class="app-container">
    <header class="app-header">
      <div class="logo">
        <h1>期货业务知识库</h1>
      </div>
      <nav class="main-nav">
        <a href="#" @click.prevent="activeTab = 'graph'" :class="{ active: activeTab === 'graph' }">知识图谱</a>
        <a href="#" @click.prevent="activeTab = 'search'" :class="{ active: activeTab === 'search' }">搜索查询</a>
        <a href="#" @click.prevent="activeTab = 'upload'" :class="{ active: activeTab === 'upload' }">文档管理</a>
      </nav>
      <div class="user-info">
        <span class="username">管理员</span>
        <div class="avatar">A</div>
      </div>
    </header>

    <main class="app-main">
      <!-- 知识图谱视图 -->
      <div v-if="activeTab === 'graph'" class="tab-content">
        <div class="content-header">
          <h2>期货业务知识图谱</h2>
          <div class="year-filter">
            <label>年份筛选:</label>
            <select v-model="activeYear" @change="changeYear">
              <option v-for="year in years" :key="year" :value="year">{{ year }}年</option>
            </select>
          </div>
        </div>
        <div class="content-body">
          <KnowledgeGraph 
            :graphData="graphData" 
            :loading="loading" 
            :error="error" 
          />
        </div>
      </div>

      <!-- 搜索查询视图 -->
      <div v-if="activeTab === 'search'" class="tab-content">
        <div class="content-header">
          <h2>期货业务知识搜索</h2>
        </div>
        <div class="content-body">
          <SearchPanel 
            :apiBaseUrl="apiBaseUrl" 
            @search-result="handleSearchResult" 
          />
        </div>
      </div>

      <!-- 文档管理视图 -->
      <div v-if="activeTab === 'upload'" class="tab-content">
        <div class="content-header">
          <h2>期货业务文档管理</h2>
        </div>
        <div class="content-body">
          <DocumentUploader 
            :apiBaseUrl="apiBaseUrl" 
            @upload-success="onUploadSuccess" 
          />
        </div>
      </div>
    </main>

    <footer class="app-footer">
      <div class="footer-content">
        <div>期货业务知识库项目 | 版本 v1.0.0</div>
        <div>© 2025 All Rights Reserved</div>
      </div>
    </footer>
  </div>
</template>

<script>
import { defineComponent, ref, onMounted } from 'vue';
import axios from 'axios';
import KnowledgeGraph from './components/KnowledgeGraph.vue';
import DocumentUploader from './components/DocumentUploader.vue';
import SearchPanel from './components/SearchPanel.vue';

export default defineComponent({
  name: 'App',
  components: {
    KnowledgeGraph,
    DocumentUploader,
    SearchPanel,
  },
  setup() {
    const activeTab = ref('graph');
    const loading = ref(false);
    const error = ref('');
    const graphData = ref(null);
    const activeYear = ref(new Date().getFullYear());
    const years = ref([]);
    const apiBaseUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';

    // 获取可用年份列表
    const fetchYears = async () => {
      try {
        const response = await axios.get(`${apiBaseUrl}/cypher`, {
          params: {
            query: "MATCH (r:RULE) RETURN DISTINCT SUBSTRING(r.effective_date, 0, 4) AS year ORDER BY year DESC"
          }
        });
        
        if (response.data && response.data.length > 0) {
          years.value = response.data.map(record => parseInt(record.year));
          // 如果没有数据，默认使用当前年份
          if (years.value.length === 0) {
            years.value = [new Date().getFullYear()];
          }
          activeYear.value = years.value[0];
        }
      } catch (err) {
        console.error('获取年份列表失败:', err);
        error.value = '获取年份列表失败，请检查API连接';
      }
    };

    // 获取图谱数据
    const fetchGraphData = async (year = null) => {
      loading.value = true;
      error.value = '';
      
      try {
        const url = `${apiBaseUrl}/graph/rule-contract`;
        const params = year ? { effective_year: year } : {};
        
        const response = await axios.get(url, { params });
        graphData.value = response.data;
      } catch (err) {
        console.error('获取图谱数据失败:', err);
        error.value = '获取图谱数据失败，请检查API连接';
        graphData.value = null;
      } finally {
        loading.value = false;
      }
    };

    // 更改年份筛选
    const changeYear = (event) => {
      activeYear.value = parseInt(event.target.value);
      fetchGraphData(activeYear.value);
    };

    // 上传文档成功后刷新数据
    const onUploadSuccess = () => {
      fetchYears();
      fetchGraphData(activeYear.value);
    };

    // 搜索结果处理
    const handleSearchResult = (result) => {
      // 处理搜索结果
      console.log('搜索结果:', result);
    };

    onMounted(() => {
      fetchYears();
      fetchGraphData();
    });

    return {
      activeTab,
      loading,
      error,
      graphData,
      activeYear,
      years,
      apiBaseUrl,
      changeYear,
      onUploadSuccess,
      handleSearchResult
    };
  },
});
</script>

<style>
/* src/styles/main.css */
:root {
  --primary-color: #3498db;
  --secondary-color: #2c3e50;
  --accent-color: #e74c3c;
  --background-color: #f5f7fa;
  --text-color: #333;
  --border-color: #ddd;
}

* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
  color: var(--text-color);
  background-color: var(--background-color);
  font-size: 14px;
  line-height: 1.5;
}

.app-container {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}

.app-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 20px;
  height: 60px;
  background-color: white;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.logo h1 {
  font-size: 20px;
  font-weight: bold;
  color: var(--primary-color);
}

.main-nav {
  display: flex;
  gap: 20px;
}

.main-nav a {
  text-decoration: none;
  color: var(--secondary-color);
  font-weight: 500;
  padding: 8px 12px;
  border-radius: 4px;
  transition: background-color 0.3s ease;
}

.main-nav a:hover {
  background-color: rgba(0, 0, 0, 0.05);
}

.main-nav a.active {
  color: var(--primary-color);
  border-bottom: 2px solid var(--primary-color);
}

.user-info {
  display: flex;
  align-items: center;
  gap: 10px;
}

.username {
  font-weight: 500;
}

.avatar {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  background-color: var(--primary-color);
  color: white;
  display: flex;
  justify-content: center;
  align-items: center;
  font-weight: bold;
}

.app-main {
  flex: 1;
  padding: 20px;
  max-width: 1400px;
  margin: 0 auto;
  width: 100%;
}

.content-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.content-header h2 {
  font-size: 22px;
  font-weight: 600;
  color: var(--secondary-color);
}

.year-filter {
  display: flex;
  align-items: center;
  gap: 10px;
}

.year-filter select {
  padding: 6px 10px;
  border: 1px solid var(--border-color);
  border-radius: 4px;
  background-color: white;
}

.content-body {
  background-color: white;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
}

.tab-content {
  animation: fadeIn 0.3s ease;
}

@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

.app-footer {
  padding: 20px;
  background-color: var(--secondary-color);
  color: white;
}

.footer-content {
  display: flex;
  justify-content: space-between;
  max-width: 1400px;
  margin: 0 auto;
  width: 100%;
  font-size: 12px;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .app-header {
    flex-direction: column;
    height: auto;
    padding: 10px;
  }
  
  .main-nav {
    margin: 10px 0;
  }
  
  .content-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 10px;
  }
  
  .footer-content {
    flex-direction: column;
    gap: 10px;
    text-align: center;
  }
}
</style>