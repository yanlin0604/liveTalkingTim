# LiveTalking Shell 脚本使用说明

## 概述

LiveTalking 提供了三个主要的 shell 脚本来管理服务：

1. **`start.sh`** - 服务启动脚本
2. **`stop.sh`** - 服务停止脚本  
3. **`status.sh`** - 服务状态检查脚本

## 脚本功能

### start.sh - 服务启动脚本

支持两种启动模式：

#### 分离式服务模式（默认）
```bash
# 启动分离式服务（主服务 + 管理服务器）
./start.sh

# 自定义端口
./start.sh --main-port 8020 --management-port 8021

# 使用指定配置文件
./start.sh --config my_config.json
```

#### 单服务模式（兼容旧版本）
```bash
# 启动单服务模式
./start.sh --single

# 自定义端口
./start.sh --single --main-port 8020
```

#### 命令行参数
- `--single`: 启动单服务模式
- `--main-port PORT`: 主服务端口（默认: 8010）
- `--management-port PORT`: 管理服务器端口（默认: 8011）
- `--config FILE`: 配置文件路径（默认: config.json）
- `--help, -h`: 显示帮助信息

### stop.sh - 服务停止脚本

```bash
# 停止所有 LiveTalking 服务
./stop.sh
```

功能：
- 通过 PID 文件优雅停止服务
- 强制停止剩余进程
- 检查端口监听状态
- 清理 PID 文件

### status.sh - 服务状态检查脚本

```bash
# 检查服务运行状态
./status.sh
```

功能：
- 检查进程运行状态
- 检查端口监听状态
- 检查 PID 文件状态
- 检查日志文件状态
- 检查服务可用性
- 显示服务状态总结

## 使用示例

### 1. 启动服务

```bash
# 启动分离式服务（推荐）
./start.sh

# 启动单服务模式
./start.sh --single

# 自定义配置启动
./start.sh --config production.json --main-port 9000 --management-port 9001
```

### 2. 检查服务状态

```bash
# 查看详细状态
./status.sh
```

输出示例：
```
>>> LiveTalking 服务状态检查
==================================
>>> 进程状态:
>>> ✅ 找到以下运行中的进程:
>>>   PID: 12345 | 命令: python app.py --config_file config.json
>>>   PID: 12346 | 命令: python management_server.py --port 8011

>>> 端口监听状态:
>>> 主服务端口 8010:
>>>   ✅ 正在监听
>>> 管理服务器端口 8011:
>>>   ✅ 正在监听

>>> 服务状态总结:
>>> 🟢 分离式服务模式运行中
```

### 3. 停止服务

```bash
# 停止所有服务
./stop.sh
```

输出示例：
```
>>> LiveTalking 服务停止脚本

>>> 停止主服务 (PID: 12345)...
>>> 主服务已停止
>>> 停止管理服务器 (PID: 12346)...
>>> 管理服务器已停止
>>> ✅ 所有服务已成功停止
```

## 日志管理

### 日志文件位置
- 主服务日志: `/mnt/disk1/ftp/data/60397193/logs/main.log`
- 管理服务器日志: `/mnt/disk1/ftp/data/60397193/logs/management.log`

### 查看日志
```bash
# 查看主服务日志
tail -f /mnt/disk1/ftp/data/60397193/logs/main.log

# 查看管理服务器日志
tail -f /mnt/disk1/ftp/data/60397193/logs/management.log

# 查看错误日志
grep ERROR /mnt/disk1/ftp/data/60397193/logs/main.log
```

## PID 文件管理

脚本会自动创建和管理 PID 文件：
- 主服务 PID: `/mnt/disk1/ftp/data/60397193/logs/main.pid`
- 管理服务器 PID: `/mnt/disk1/ftp/data/60397193/logs/management.pid`

## 故障排除

### 常见问题

1. **端口被占用**
   ```bash
   # 检查端口占用
   netstat -tlnp | grep :8010
   netstat -tlnp | grep :8011
   
   # 使用不同端口启动
   ./start.sh --main-port 8020 --management-port 8021
   ```

2. **进程无法停止**
   ```bash
   # 强制停止所有相关进程
   pkill -f "app.py|management_server.py"
   
   # 或者使用 stop.sh
   ./stop.sh
   ```

3. **配置文件不存在**
   ```bash
   # 创建默认配置文件
   python run_dynamic.py --create-config
   
   # 然后启动服务
   ./start.sh
   ```

4. **权限问题**
   ```bash
   # 给脚本添加执行权限
   chmod +x start.sh stop.sh status.sh
   ```

### 调试模式

```bash
# 查看详细启动信息
bash -x start.sh

# 查看详细停止信息
bash -x stop.sh
```

## 自动化部署

### systemd 服务文件

创建 `/etc/systemd/system/livetalking.service`:

```ini
[Unit]
Description=LiveTalking Digital Human Service
After=network.target

[Service]
Type=forking
User=your_user
WorkingDirectory=/path/to/livetalking
ExecStart=/path/to/livetalking/start.sh
ExecStop=/path/to/livetalking/stop.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

启用服务：
```bash
sudo systemctl enable livetalking
sudo systemctl start livetalking
sudo systemctl status livetalking
```

### crontab 自动重启

```bash
# 编辑 crontab
crontab -e

# 添加定时重启（每天凌晨2点重启）
0 2 * * * /path/to/livetalking/stop.sh && sleep 5 && /path/to/livetalking/start.sh
```

## 监控和告警

### 简单的监控脚本

创建 `monitor.sh`:

```bash
#!/bin/bash
# 检查服务状态，如果失败则重启

if ! ./status.sh | grep -q "🟢 分离式服务模式运行中"; then
    echo "$(date): 服务异常，正在重启..."
    ./stop.sh
    sleep 5
    ./start.sh
    echo "$(date): 服务重启完成"
fi
```

添加到 crontab：
```bash
# 每5分钟检查一次
*/5 * * * * /path/to/livetalking/monitor.sh >> /path/to/livetalking/monitor.log 2>&1
```

## 注意事项

1. **路径配置**: 确保脚本中的路径配置正确
2. **权限设置**: 确保有读写日志目录的权限
3. **环境变量**: 确保 conda 环境正确激活
4. **端口冲突**: 避免与其他服务端口冲突
5. **日志轮转**: 定期清理日志文件避免磁盘空间不足

## 最佳实践

1. **生产环境**: 使用分离式服务模式
2. **开发环境**: 可以使用单服务模式便于调试
3. **日志管理**: 定期检查日志文件大小和内容
4. **监控**: 设置服务监控和自动重启
5. **备份**: 定期备份配置文件和重要数据 