# 容器与 Kubernetes 访问权限（Container / K8s Access）

## 1. 集群与命名空间划分
- 按环境划分集群/命名空间：dev / staging / production
- 生产命名空间默认对研发只读，写入/exec 权限需单独申请

## 2. 权限申请流程
- 提交权限申请：目标集群、命名空间、权限级别（view/edit/admin）、期限
- 生产环境的 `edit`/`exec` 权限走两级审批：技术负责人 + SRE Owner
- 通过 RBAC RoleBinding 自动下发，到期自动回收（默认 7/14/30 天）

## 3. 常见操作与限制
- 允许：查看 Pod 状态、日志（`kubectl logs`）、只读描述资源
- 受限：`kubectl exec` 进入生产容器、修改 Deployment/ConfigMap，需要临时授权且记录审计
- 禁止：直接删除生产命名空间下的关键资源（需通过发布流水线操作）

## 4. Secrets 与配置
- 生产集群的 Secrets 由密钥管理系统统一下发，禁止手工 `kubectl create secret` 明文写入
- ConfigMap 中不允许包含密码、Token 等敏感信息

## 5. 审计与异常
- 所有 `exec`/`edit` 操作记录审计日志，定期抽查
- 发现异常操作（非工作时间的生产 exec、权限组外账号操作）需按事故响应流程升级
