#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Log Reader Web Interface
基于Flask的日志文件读取Web界面
"""

from flask import Flask, render_template, request, jsonify
import os
import json
from datetime import datetime
from log_reader import LogReader

app = Flask(__name__)

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

if __name__ == '__main__':
    # 创建templates目录
    os.makedirs('templates', exist_ok=True)
    
    print("启动日志读取器Web界面...")
    print("访问地址: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
