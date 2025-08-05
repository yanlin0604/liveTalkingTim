#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Log Reader Web Interface
基于Flask的日志文件读取Web界面
"""

from flask import Flask, render_template, request, jsonify, Response
import os
import json
import time
import threading
from datetime import datetime
from log_reader import LogReader

app = Flask(__name__)

# 存储活跃的实时读取会话
active_sessions = {}

@app.route('/')
def index():
    """主页面"""
    return render_template('log_reader.html')

@app.route('/api/read_log', methods=['POST'])
def read_log():
    """读取日志文件API"""
    try:
        data = request.get_json()
        file_path = data.get('file_path', '')
        num_lines = data.get('num_lines', 50)
        from_end = data.get('from_end', True)
        
        if not file_path:
            return jsonify({
                'success': False,
                'error': '请提供文件路径'
            })
        
        # 创建日志读取器
        reader = LogReader(file_path)
        
        # 获取文件信息
        file_info = reader.get_file_info()
        if not file_info:
            return jsonify({
                'success': False,
                'error': '文件不存在或无法访问'
            })
        
        # 读取日志内容
        if num_lines == -1:  # -1 表示读取全部
            lines = reader.read_lines(num_lines=None)
        else:
            lines = reader.read_lines(num_lines=num_lines, from_end=from_end)
        
        return jsonify({
            'success': True,
            'data': {
                'lines': lines,
                'total_lines': len(lines),
                'file_info': {
                    'path': file_info['path'],
                    'size': file_info['size'],
                    'modified': datetime.fromtimestamp(file_info['modified']).strftime('%Y-%m-%d %H:%M:%S')
                }
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'读取文件时出错: {str(e)}'
        })

@app.route('/api/validate_path', methods=['POST'])
def validate_path():
    """验证文件路径API"""
    try:
        data = request.get_json()
        file_path = data.get('file_path', '')
        
        if not file_path:
            return jsonify({
                'valid': False,
                'error': '请提供文件路径'
            })
        
        reader = LogReader(file_path)
        file_info = reader.get_file_info()
        
        if file_info:
            return jsonify({
                'valid': True,
                'info': {
                    'size': file_info['size'],
                    'modified': datetime.fromtimestamp(file_info['modified']).strftime('%Y-%m-%d %H:%M:%S')
                }
            })
        else:
            return jsonify({
                'valid': False,
                'error': '文件不存在或无法访问'
            })
            
    except Exception as e:
        return jsonify({
            'valid': False,
            'error': f'验证路径时出错: {str(e)}'
        })

@app.route('/api/start_realtime', methods=['POST'])
def start_realtime():
    """启动实时日志读取"""
    try:
        data = request.get_json()
        file_path = data.get('file_path', '')
        interval = data.get('interval', 1.0)
        initial_lines = data.get('initial_lines', 10)
        
        if not file_path:
            return jsonify({
                'success': False,
                'error': '请提供文件路径'
            })
        
        reader = LogReader(file_path)
        file_info = reader.get_file_info()
        if not file_info:
            return jsonify({
                'success': False,
                'error': '文件不存在或无法访问'
            })
        
        # 生成会话ID
        session_id = f"{int(time.time())}_{hash(file_path)}"
        
        # 获取初始内容
        initial_content = reader.get_recent_lines(initial_lines)
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'initial_content': initial_content,
            'file_info': {
                'path': file_info['path'],
                'size': file_info['size'],
                'modified': datetime.fromtimestamp(file_info['modified']).strftime('%Y-%m-%d %H:%M:%S')
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'启动实时读取时出错: {str(e)}'
        })

@app.route('/api/realtime_stream/<session_id>')
def realtime_stream(session_id):
    """实时日志流 (Server-Sent Events)"""
    def generate():
        try:
            # 从会话参数中获取配置
            file_path = request.args.get('file_path')
            interval = float(request.args.get('interval', 1.0))
            
            if not file_path:
                yield f"data: {{\"error\": \"缺少文件路径\"}}\n\n"
                return
            
            reader = LogReader(file_path)
            if not reader.validate_file():
                yield f"data: {{\"error\": \"文件不存在或无法访问\"}}\n\n"
                return
            
            # 创建停止事件
            stop_event = threading.Event()
            active_sessions[session_id] = stop_event
            
            # 发送连接成功消息
            yield f"data: {{\"type\": \"connected\", \"message\": \"实时日志连接已建立\"}}\n\n"
            
            def send_line(line):
                timestamp = datetime.now().strftime('%H:%M:%S')
                line_data = {
                    'type': 'log_line',
                    'timestamp': timestamp,
                    'content': line
                }
                return f"data: {json.dumps(line_data, ensure_ascii=False)}\n\n"
            
            # 使用队列来收集新行
            line_queue = []
            
            def line_callback(line):
                line_queue.append(send_line(line))
            
            # 启动跟踪线程
            follow_thread = threading.Thread(
                target=reader.tail_follow,
                args=(line_callback, interval, stop_event)
            )
            follow_thread.daemon = True
            follow_thread.start()
            
            # 持续发送新行
            while not stop_event.is_set():
                if line_queue:
                    for line_data in line_queue:
                        yield line_data
                    line_queue.clear()
                else:
                    time.sleep(0.1)
                    
        except Exception as e:
            yield f"data: {{\"error\": \"实时读取出错: {str(e)}\"}}\n\n"
        finally:
            # 清理会话
            if session_id in active_sessions:
                del active_sessions[session_id]
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/stop_realtime/<session_id>', methods=['POST'])
def stop_realtime(session_id):
    """停止实时日志读取"""
    try:
        if session_id in active_sessions:
            active_sessions[session_id].set()
            del active_sessions[session_id]
            return jsonify({'success': True, 'message': '实时读取已停止'})
        else:
            return jsonify({'success': False, 'error': '会话不存在'})
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'停止实时读取时出错: {str(e)}'
        })

if __name__ == '__main__':
    # 创建templates目录
    os.makedirs('templates', exist_ok=True)
    
    print("启动日志读取器Web界面...")
    print("访问地址: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
