# frontend/Dockerfile
# 构建阶段
FROM node:16-alpine as build-stage

# 设置工作目录
WORKDIR /app

# 复制 package.json 和 package-lock.json
COPY package*.json ./

# 安装项目依赖
RUN npm install

# 创建基本的 vite.config.js 文件
RUN echo 'import { defineConfig } from "vite"; \
import vue from "@vitejs/plugin-vue"; \
export default defineConfig({ \
  plugins: [vue()], \
  root: ".", \
});' > vite.config.js

# 创建 index.html 文件
RUN echo '<!DOCTYPE html>\
<html lang="en">\
<head>\
  <meta charset="UTF-8">\
  <meta name="viewport" content="width=device-width, initial-scale=1.0">\
  <title>期货业务知识库</title>\
</head>\
<body>\
  <div id="app"></div>\
  <script type="module" src="/src/main.js"></script>\
</body>\
</html>' > index.html

# 复制项目文件
COPY . .

# 构建项目
RUN npm run build

# 生产阶段
FROM nginx:stable-alpine as production-stage

# 复制构建的静态文件
COPY --from=build-stage /app/dist /usr/share/nginx/html

# 复制Nginx配置文件
COPY nginx.conf /etc/nginx/conf.d/default.conf

# 暴露端口
EXPOSE 80

# 启动Nginx
CMD ["nginx", "-g", "daemon off;"]