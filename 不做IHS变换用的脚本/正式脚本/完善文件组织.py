"""
检查和完善文件组织结构
"""
import os
import shutil
from pathlib import Path

OUTPUT_DIR = r"c:\Users\29106\OneDrive\文档\毕业论文\不做IHS变换用的脚本\正式脚本\输出文件"
MAIN_DIR = r"c:\Users\29106\OneDrive\文档\毕业论文\不做IHS变换用的脚本\正式脚本\观测的辅助数据"

# 确保目录存在
Path(MAIN_DIR).mkdir(parents=True, exist_ok=True)
Path(f"{MAIN_DIR}/3_假说验证核心表格").mkdir(parents=True, exist_ok=True)

# 列出输出文件夹中的所有文件
print("输出文件夹中的文件：")
for f in os.listdir(OUTPUT_DIR):
    if f.endswith(('.csv', '.png', '.md')):
        print(f"  - {f}")

# 查找包含特殊字符的文件
print("\n查找遗漏的文件...")
target_files = {
    '规则敞口→RQ→韧性传导链_实证证据表.csv': '3_假说验证核心表格',
    '假说3验证_制度升级倒逼机制.png': '3_假说验证核心表格',
    '论文假说_综合验证仪表板.md': '3_假说验证核心表格'
}

for fname, target_folder in target_files.items():
    source_path = os.path.join(OUTPUT_DIR, fname)
    if os.path.exists(source_path):
        dest_path = os.path.join(MAIN_DIR, target_folder, fname)
        shutil.move(source_path, dest_path)
        print(f"✅ 移动: {fname}")
    else:
        print(f"❌ 找不到: {fname}")

# 输出最终结构
print("\n\n" + "="*60)
print("最终文件结构（观测的辅助数据）：")
print("="*60)

for root, dirs, files in os.walk(MAIN_DIR):
    level = root.replace(MAIN_DIR, '').count(os.sep)
    indent = ' ' * 2 * level
    folder_name = os.path.basename(root)
    if not folder_name:
        folder_name = "观测的辅助数据/"
    print(f'{indent}{folder_name}/')
    
    subindent = ' ' * 2 * (level + 1)
    for file in sorted(files):
        print(f'{subindent}├─ {file}')

print("\n✅ 文件组织完成！")
