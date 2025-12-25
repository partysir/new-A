"""
一键修复脚本
自动修复 AttributeError: 'ModelConfig' object has no attribute 'get'

运行: python quick_fix.py
"""

import re
from pathlib import Path
from datetime import datetime


def backup_file(filepath):
    """备份文件"""
    backup_path = Path(str(filepath) + f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    content = filepath.read_text(encoding='utf-8')
    backup_path.write_text(content, encoding='utf-8')
    return backup_path


def fix_ml_model():
    """修复 ml_model.py"""
    print("\n" + "=" * 60)
    print("修复 ml_model.py")
    print("=" * 60)

    ml_path = Path('ml_model.py')
    if not ml_path.exists():
        print("❌ 未找到 ml_model.py")
        return False

    # 备份
    backup_path = backup_file(ml_path)
    print(f"✓ 已备份到: {backup_path}")

    # 读取
    content = ml_path.read_text(encoding='utf-8')
    original_content = content

    # 修复所有 config.xxx.get() 调用
    fixes = [
        (r"config\.model\.get\('([^']+)',\s*([^)]+)\)", r"getattr(config.model, '\1', \2)"),
        (r"config\.label\.get\('([^']+)',\s*([^)]+)\)", r"getattr(config.label, '\1', \2)"),
        (r"config\.strategy\.get\('([^']+)',\s*([^)]+)\)", r"getattr(config.strategy, '\1', \2)"),
    ]

    for pattern, replacement in fixes:
        content = re.sub(pattern, replacement, content)

    if content != original_content:
        ml_path.write_text(content, encoding='utf-8')
        print("✅ ml_model.py 已修复")

        # 显示修改
        lines_changed = sum(1 for a, b in zip(original_content.split('\n'), content.split('\n')) if a != b)
        print(f"   修改了 {lines_changed} 行")
        return True
    else:
        print("⚠️  未找到需要修复的内容")
        return False


def fix_config():
    """修复 config.py - 添加 ConfigMixin"""
    print("\n" + "=" * 60)
    print("修复 config.py")
    print("=" * 60)

    config_path = Path('config.py')
    if not config_path.exists():
        print("❌ 未找到 config.py")
        return False

    # 备份
    backup_path = backup_file(config_path)
    print(f"✓ 已备份到: {backup_path}")

    # 读取
    content = config_path.read_text(encoding='utf-8')

    # 检查是否已经有 ConfigMixin
    if 'class ConfigMixin' in content:
        print("⚠️  ConfigMixin 已存在，跳过")
        return True

    # 添加 ConfigMixin
    mixin_code = '''

class ConfigMixin:
    """配置类混入，提供字典式访问"""

    def get(self, key: str, default=None):
        """支持字典式访问"""
        return getattr(self, key, default)

    def __getitem__(self, key: str):
        """支持 config['key'] 访问"""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"'{key}' not found")

'''

    # 在第一个 @dataclass 前插入
    match = re.search(r'@dataclass', content)
    if match:
        pos = match.start()
        content = content[:pos] + mixin_code + content[pos:]
        print("✓ 已添加 ConfigMixin 类")

    # 为所有配置类添加继承
    pattern = r'(@dataclass)\s+class\s+(\w+Config):'
    replacement = r'\1\nclass \2(ConfigMixin):'

    original = content
    content = re.sub(pattern, replacement, content)

    if content != original:
        config_path.write_text(content, encoding='utf-8')
        print("✅ config.py 已修复")
        print("   所有配置类现在支持 .get() 方法")
        return True
    else:
        print("⚠️  未找到配置类定义")
        return False


def verify_fix():
    """验证修复"""
    print("\n" + "=" * 60)
    print("验证修复")
    print("=" * 60)

    try:
        # 清除旧的导入缓存
        import sys
        if 'config' in sys.modules:
            del sys.modules['config']
        if 'ml_model' in sys.modules:
            del sys.modules['ml_model']

        # 重新导入
        from config import Config

        config = Config()

        # 测试 get 方法
        test1 = config.model.get('use_ensemble', False)
        print(f"✓ config.model.get('use_ensemble') = {test1}")

        test2 = getattr(config.model, 'forward_return_days', 5)
        print(f"✓ getattr(config.model, 'forward_return_days') = {test2}")

        print("✅ 验证成功！")
        return True

    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 70)
    print(" " * 20 + "🔧 一键修复脚本")
    print("=" * 70)
    print("\n问题: AttributeError: 'ModelConfig' object has no attribute 'get'")
    print("解决: 修复 ml_model.py 和 config.py\n")

    choice = input("开始修复? (y/n): ")

    if choice.lower() != 'y':
        print("已取消")
        return

    # 修复 ml_model.py
    ml_fixed = fix_ml_model()

    # 修复 config.py
    config_fixed = fix_config()

    # 验证
    if ml_fixed or config_fixed:
        verify_fix()

        print("\n" + "=" * 70)
        print("✅ 修复完成！")
        print("=" * 70)
        print("\n现在可以运行:")
        print("  python main.py --mode backtest")
        print("\n如果还有问题，检查备份文件:")
        print("  ml_model.py.backup_*")
        print("  config.py.backup_*")
        print("=" * 70 + "\n")
    else:
        print("\n⚠️  没有进行任何修复")


if __name__ == '__main__':
    main()