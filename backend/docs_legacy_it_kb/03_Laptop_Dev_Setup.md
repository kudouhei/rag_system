# 研发电脑与开发环境配置（macOS 优先）

## 1. 基础软件（必须）
- 密码管理器（公司指定）
- 安全代理/EDR（公司指定）
- 浏览器（Chrome/Edge）
- VPN / ZTNA 客户端

## 2. 开发工具（推荐）
- Git
- Docker Desktop / Colima（视公司策略）
- Node.js（建议使用 nvm）
- Python（建议使用 pyenv/conda，项目内用 venv/poetry 也可）
- IDE：VSCode / JetBrains

## 3. Git 访问（最常见问题）
### 3.1 HTTPS vs SSH
- SSH 更稳定：需要将公钥上传到 Git 平台
- HTTPS：需要 PAT（个人访问令牌），不建议用账号密码

### 3.2 常见报错
- `Permission denied (publickey)`：SSH key 未配置/未加到 agent/未授权
- `Repository not found`：URL 错/没有权限/组织策略限制

## 4. 本地服务运行（通用建议）
1. 优先使用 `make dev` 或 `./start.sh` 一键启动脚本（如果项目提供）
2. 端口占用：使用 `lsof -i :<port>` 查占用进程
3. 环境变量：不要把密钥写进 git；使用 `.env` 或 secrets 管理

