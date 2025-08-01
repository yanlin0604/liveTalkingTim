"""
Swagger/OpenAPI文档配置
"""
from aiohttp import web

try:
    from aiohttp_swagger3 import SwaggerDocs, SwaggerUiSettings
    SWAGGER3_AVAILABLE = True
except ImportError:
    try:
        from aiohttp_swagger import setup_swagger
        SWAGGER3_AVAILABLE = False
    except ImportError:
        print("Warning: No swagger package found. Swagger documentation will be disabled.")
        SWAGGER3_AVAILABLE = None


def create_swagger_docs(app):
    """创建Swagger文档"""
    
    if SWAGGER3_AVAILABLE is None:
        print("Swagger documentation disabled - no swagger package available")
        return app
    
    if SWAGGER3_AVAILABLE:
        # 使用 aiohttp-swagger3
        return create_swagger3_docs(app)
    else:
        # 使用 aiohttp-swagger (旧版本)
        return create_swagger2_docs(app)

def create_swagger3_docs(app):
    """使用 aiohttp-swagger3 创建文档"""
    try:
        from aiohttp_swagger3 import SwaggerDocs, SwaggerUiSettings
        
        # 创建Swagger文档实例
        swagger = SwaggerDocs(
            app,
            swagger_ui_settings=SwaggerUiSettings(path="/swagger"),
            title="LiveTalking API",
            version="1.0.0",
            description="LiveTalking数字人实时对话系统API文档"
        )
        
        print("Swagger 3.0 documentation enabled at /swagger")
        return app
        
    except Exception as e:
        print(f"Failed to setup Swagger 3.0: {e}")
        return app

def create_swagger2_docs(app):
    """使用 aiohttp-swagger 创建文档 (旧版本)"""
    
    # 设置Swagger文档 - 使用自动扫描模式
    setup_swagger(
        app,
        swagger_url="/swagger",
        title="LiveTalking API",
        description="LiveTalking数字人实时对话系统API文档",
        api_version="1.0.0",
        contact="lipku@foxmail.com"
    )
    
    print("Swagger 2.0 documentation enabled at /swagger")
    return app 