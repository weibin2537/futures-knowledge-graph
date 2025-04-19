// src/components/DocumentUploader.vue
<template>
  <div class="document-uploader">
    <div class="uploader-header">
      <h2>文档上传</h2>
      <p class="description">
        支持上传PDF、Word、Excel和Markdown格式的期货业务文档
      </p>
    </div>

    <div 
      class="upload-zone" 
      @dragover.prevent="dragover" 
      @dragleave.prevent="dragleave" 
      @drop.prevent="drop"
      :class="{ 'active': isDragging }"
    >
      <div v-if="uploadProgress > 0 && uploadProgress < 100" class="progress-container">
        <div class="progress-bar" :style="{ width: `${uploadProgress}%` }"></div>
        <span class="progress-text">上传中 ({{ uploadProgress }}%)</span>
      </div>
      <div v-else-if="isUploading" class="processing">
        <div class="spinner"></div>
        <span>文档处理中...</span>
      </div>
      <div v-else class="upload-content">
        <div class="upload-icon">
          <i class="icon-upload"></i>
        </div>
        <p>拖放文件到此处，或</p>
        <label class="file-select-btn">
          选择文件
          <input 
            type="file" 
            @change="handleFileSelect" 
            accept=".pdf,.docx,.doc,.xlsx,.xls,.md,.markdown"
            style="display: none;"
          />
        </label>
      </div>
    </div>

    <div v-if="error" class="error-message">
      {{ error }}
    </div>

    <div v-if="recentUploads.length > 0" class="recent-uploads">
      <h3>最近上传</h3>
      <table>
        <thead>
          <tr>
            <th>文件名</th>
            <th>类型</th>
            <th>上传时间</th>
            <th>实体数量</th>
            <th>操作</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(doc, index) in recentUploads" :key="index">
            <td>{{ doc.metadata.file_name }}</td>
            <td>{{ formatDocType(doc.metadata.file_type) }}</td>
            <td>{{ formatDate(doc.metadata.import_time) }}</td>
            <td>{{ doc.entity_count || 0 }}</td>
            <td>
              <button class="view-btn" @click="viewDocument(doc.doc_id)">查看</button>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<script>
import { defineComponent, ref, onMounted } from 'vue';
import axios from 'axios';

export default defineComponent({
  name: 'DocumentUploader',
  emits: ['upload-success'],
  props: {
    apiBaseUrl: {
      type: String,
      default: 'http://localhost:8000'
    }
  },
  setup(props, { emit }) {
    const isDragging = ref(false);
    const isUploading = ref(false);
    const uploadProgress = ref(0);
    const error = ref('');
    const recentUploads = ref([]);

    // 获取最近上传的文档
    const fetchRecentUploads = async () => {
      try {
        // 此处应该调用API获取最近上传的文档列表
        // 但因为API中没有这个接口，所以我们先使用模拟数据
        // 实际项目中应添加一个获取最近上传文档的API
        
        // 模拟获取最近上传的文档
        const response = await axios.get(`${props.apiBaseUrl}/cypher`, {
          params: {
            query: `
              MATCH (d:DOCUMENT)
              RETURN d.doc_id AS doc_id, d.file_name AS file_name, 
                     d.file_type AS file_type, d.import_time AS import_time,
                     d.page_count AS page_count
              ORDER BY d.import_time DESC
              LIMIT 5
            `
          }
        });
        
        if (response.data && response.data.length > 0) {
          recentUploads.value = response.data.map(doc => ({
            doc_id: doc.doc_id,
            metadata: {
              file_name: doc.file_name,
              file_type: doc.file_type,
              import_time: doc.import_time,
              page_count: doc.page_count
            },
            entity_count: 0 // 实际项目中应从API获取
          }));
        }
      } catch (err) {
        console.error('获取最近上传文档失败:', err);
        error.value = '获取最近上传文档失败';
      }
    };

    // 处理文件拖拽
    const dragover = (e) => {
      isDragging.value = true;
    };

    const dragleave = (e) => {
      isDragging.value = false;
    };

    const drop = (e) => {
      isDragging.value = false;
      const files = e.dataTransfer.files;
      if (files.length > 0) {
        uploadFile(files[0]);
      }
    };

    // 处理文件选择
    const handleFileSelect = (e) => {
      const files = e.target.files;
      if (files.length > 0) {
        uploadFile(files[0]);
      }
    };

    // 上传文件
    const uploadFile = async (file) => {
      // 验证文件类型
      const validTypes = [
        'application/pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/msword',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'application/vnd.ms-excel',
        'text/markdown',
        'text/plain'
      ];

      if (!validTypes.includes(file.type) && !file.name.endsWith('.md')) {
        error.value = '不支持的文件类型，请上传PDF、Word、Excel或Markdown文件';
        return;
      }

      isUploading.value = true;
      uploadProgress.value = 0;
      error.value = '';

      try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await axios.post(
          `${props.apiBaseUrl}/documents/upload`,
          formData,
          {
            headers: {
              'Content-Type': 'multipart/form-data'
            },
            onUploadProgress: (progressEvent) => {
              uploadProgress.value = Math.round(
                (progressEvent.loaded * 100) / progressEvent.total
              );
            }
          }
        );

        // 上传成功
        if (response.data) {
          // 添加到最近上传列表
          recentUploads.value.unshift({
            doc_id: response.data.doc_id,
            metadata: response.data.metadata,
            entity_count: response.data.entity_count || 0
          });

          // 限制最近上传列表长度
          if (recentUploads.value.length > 5) {
            recentUploads.value = recentUploads.value.slice(0, 5);
          }

          // 触发上传成功事件
          emit('upload-success', response.data);
        }
      } catch (err) {
        console.error('文档上传失败:', err);
        error.value = err.response?.data?.detail || '文档上传失败，请重试';
      } finally {
        isUploading.value = false;
        uploadProgress.value = 0;
      }
    };

    // 查看文档
    const viewDocument = (docId) => {
      // 在实际项目中，这里应该跳转到文档详情页面
      // 或者打开文档预览模态框
      console.log('查看文档:', docId);
      // 可以实现为打开一个模态框，显示文档的详细信息和提取的实体
    };

    // 格式化文档类型
    const formatDocType = (type) => {
      const typeMap = {
        'pdf': 'PDF',
        'docx': 'Word',
        'doc': 'Word',
        'xlsx': 'Excel',
        'xls': 'Excel',
        'markdown': 'Markdown',
        'md': 'Markdown'
      };
      return typeMap[type] || type;
    };

    // 格式化日期
    const formatDate = (dateString) => {
      if (!dateString) return '-';
      try {
        const date = new Date(dateString);
        return `${date.getFullYear()}-${(date.getMonth() + 1).toString().padStart(2, '0')}-${date.getDate().toString().padStart(2, '0')} ${date.getHours().toString().padStart(2, '0')}:${date.getMinutes().toString().padStart(2, '0')}`;
      } catch (err) {
        return dateString;
      }
    };

    onMounted(() => {
      fetchRecentUploads();
    });

    return {
      isDragging,
      isUploading,
      uploadProgress,
      error,
      recentUploads,
      dragover,
      dragleave,
      drop,
      handleFileSelect,
      viewDocument,
      formatDocType,
      formatDate
    };
  }
});
</script>

<style scoped>
.document-uploader {
  padding: 20px;
  background-color: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
}

.uploader-header {
  margin-bottom: 20px;
}

.uploader-header h2 {
  margin: 0 0 8px 0;
  font-size: 20px;
}

.description {
  margin: 0;
  color: #666;
}

.upload-zone {
  position: relative;
  min-height: 200px;
  border: 2px dashed #ccc;
  border-radius: 8px;
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 20px;
  margin-bottom: 20px;
  transition: all 0.3s ease;
}

.upload-zone.active {
  border-color: #2980b9;
  background-color: rgba(41, 128, 185, 0.05);
}

.upload-content {
  text-align: center;
}

.upload-icon {
  font-size: 48px;
  color: #7f8c8d;
  margin-bottom: 16px;
}

.icon-upload:before {
  content: "↑";
}

.file-select-btn {
  display: inline-block;
  padding: 8px 16px;
  background-color: #3498db;
  color: white;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.3s ease;
}

.file-select-btn:hover {
  background-color: #2980b9;
}

.progress-container {
  width: 100%;
  height: 20px;
  background-color: #f0f0f0;
  border-radius: 10px;
  overflow: hidden;
  position: relative;
}

.progress-bar {
  height: 100%;
  background-color: #2ecc71;
  transition: width 0.3s ease;
}

.progress-text {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  text-align: center;
  line-height: 20px;
  color: #333;
  font-size: 12px;
}

.processing {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
}

.spinner {
  width: 30px;
  height: 30px;
  border: 3px solid #f3f3f3;
  border-top: 3px solid #3498db;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.error-message {
  padding: 12px;
  background-color: #ffecec;
  color: #e74c3c;
  border-radius: 4px;
  margin-bottom: 20px;
}

.recent-uploads {
  margin-top: 30px;
}

.recent-uploads h3 {
  margin-top: 0;
  margin-bottom: 12px;
  font-size: 16px;
}

table {
  width: 100%;
  border-collapse: collapse;
}

th, td {
  padding: 10px;
  text-align: left;
  border-bottom: 1px solid #eee;
}

th {
  background-color: #f9f9f9;
  font-weight: 600;
}

.view-btn {
  padding: 4px 8px;
  background-color: #3498db;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.view-btn:hover {
  background-color: #2980b9;
}
</style>