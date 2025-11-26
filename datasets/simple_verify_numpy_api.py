import os
import sys

# 设置sys.path，确保能找到preprocess模块
current_file = os.path.abspath(__file__)
# 向上跳三级：datasets → preprocess → 根目录
root_path = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
sys.path.append(root_path)

# 导入基础模块
import requests
import json
import numpy as np
import cv2
from preprocess.geometry.lineseg import LineSegment, LineSegments
from PyQt5.QtWidgets import QApplication

# API URL
API_URL = "http://127.0.0.1:8000/detect/once"

# 测试用PCD文件路径
TEST_PCD_FILE = r"C:\Users\Server\xwechat_files\wxid_649z3170r5rz22_2353\msg\file\2025-11\Preview10.27_5_noreflect.pcd"

# 配置文件路径
CFG_PATH = r"D:\work\l2bim\configs\interval\15m\1F\1f_office_03.yaml"

print("=== 简单验证detect/once接口返回numpy数组功能 ===")

# 验证测试文件存在
if not os.path.exists(TEST_PCD_FILE):
    print(f"错误: 测试文件不存在: {TEST_PCD_FILE}")
    exit(1)

# 调用API并获取numpy数组
with open(TEST_PCD_FILE, 'rb') as pcd_file:
    files = {'file': pcd_file}
    data = {'cfg': CFG_PATH, 'return_numpy': True, 'return_linesegs': True}
    print(f"📤 发送请求: 文件={TEST_PCD_FILE}, return_numpy=true, return_linesegs=true")
    
    response = requests.post(API_URL, files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ 请求成功!")
        print(f"✅ 响应状态: {result.get('message')}")
        
        # 检查img字段
        img_data = result.get('img')
        linesegs_data = result.get('serialized_linesegs')
        if img_data is not None:
            print(f"✅ 成功获取numpy数组图像数据!")
            
            # 转换为numpy数组
            img_array = np.array(img_data)
            print(f"✅ 成功转换为numpy数组!")
            print(f"✅ 图像形状: {img_array.shape}")
            print(f"✅ 图像数据类型: {img_array.dtype}")
            
            # 转换图像数据类型为uint8以便显示
            if img_array.dtype != np.uint8:
                print(f"🔄 转换图像数据类型从 {img_array.dtype} 到 uint8")
                img_array = ((img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8) * 255).astype(np.uint8)

            if linesegs_data is not None:
                print(f"✅ 成功获取线段数据!")
                print(f"✅ 包含{len(linesegs_data)}条线段")
                
                # 将序列化的线段数据转换为LineSegments对象
                line_objects = []
                for seg_data in linesegs_data:
                    point_a = np.array(seg_data['point_a'])
                    point_b = np.array(seg_data['point_b'])
                    line_seg = LineSegment(point_a, point_b)
                    line_objects.append(line_seg)
                
                # 创建LineSegments对象
                linesegs_obj = LineSegments(line_objects)
                print(f"✅ 成功创建LineSegments对象，包含{len(linesegs_obj.linesegments)}条线段")
            else:
                print(f"❌ 未获取到线段数据!")
                print(f"🔄 创建测试线条数据...")
                # 创建测试线条
                line1 = LineSegment(np.array([100, 100]), np.array([400, 100]))  # 屋顶线
                line2 = LineSegment(np.array([100, 100]), np.array([100, 400]))  # 左边线
                line3 = LineSegment(np.array([400, 100]), np.array([400, 400]))  # 右边线
                line4 = LineSegment(np.array([100, 400]), np.array([400, 400]))  # 底线
                line5 = LineSegment(np.array([450, 150]), np.array([600, 150]))  # 小屋顶线
                line6 = LineSegment(np.array([450, 150]), np.array([450, 350]))  # 小左边线
                line7 = LineSegment(np.array([600, 150]), np.array([600, 350]))  # 小右边线
                line8 = LineSegment(np.array([450, 350]), np.array([600, 350]))  # 小底线
                linesegs_obj = LineSegments([line1, line2, line3, line4, line5, line6, line7, line8])
                print(f"✅ 成功创建测试LineSegments对象，包含{len(linesegs_obj.linesegments)}条线段")
            
            # 创建QApplication实例
            app = QApplication(sys.argv)
            
            # 在QApplication创建后再导入TestWindow
            from preprocess.datasets.line_editor_qt import TestWindow
            
            # 设置全局样式表
            app.setStyleSheet("""
                QWidget {
                    background-color: #2d2d2d;
                    color: #ffffff;
                    font-family: Arial, sans-serif;
                }
                QMainWindow, QWidget {
                    border: none;
                }
            """)
            
            
            # 创建窗口并显示
            print("🖼️  正在显示图像窗口...")

            window = TestWindow(img=img_array, linesegs=linesegs_obj)
            window.show()
            
            # 运行Qt应用程序
            sys.exit(app.exec_())
        else:
            print("❌ 未返回numpy数组图像数据")

        
    else:
        print(f"❌ 请求失败，状态码: {response.status_code}")
        print(f"❌ 错误信息: {response.text}")