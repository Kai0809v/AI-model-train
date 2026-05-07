"""
将 PNG 图标转换为 ICO 格式（多尺寸）
用于 PyInstaller 打包
"""
from PIL import Image
import os

def png_to_ico(png_path, ico_path):
    """
    将 PNG 转换为多尺寸 ICO 文件
    
    支持的尺寸: 256x256, 128x128, 64x64, 48x48, 32x32, 16x16
    """
    if not os.path.exists(png_path):
        print(f"错误: 找不到文件 {png_path}")
        return False
    
    try:
        # 打开 PNG 图片
        img = Image.open(png_path)
        
        # 定义需要生成的尺寸
        sizes = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)]
        
        # 调整大小并保存为 ICO
        img.save(ico_path, format='ICO', sizes=sizes)
        
        print(f"✓ 成功转换: {png_path} -> {ico_path}")
        print(f"  包含尺寸: {', '.join([f'{w}x{h}' for w, h in sizes])}")
        return True
        
    except Exception as e:
        print(f"✗ 转换失败: {e}")
        return False

if __name__ == "__main__":
    png_file = "res/icon.png"
    ico_file = "res/icon.ico"
    
    print("=" * 50)
    print("PNG to ICO 转换器")
    print("=" * 50)
    
    if png_to_ico(png_file, ico_file):
        print("\n完成！现在可以在 APredict.spec 中使用 icon.ico")
    else:
        print("\n失败！请检查错误信息")
