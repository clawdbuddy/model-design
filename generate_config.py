#!/usr/bin/env python3
"""
generate_config.py - 用 Token API 生成模型设计配置
"""
import argparse
import json
import os
import sys

try:
    from openai import OpenAI
except ImportError:
    print("请先安装: pip install openai")
    sys.exit(1)


def generate_config_with_api(requirement: str, api_key: str, model: str = "gpt-4"):
    """调用 Token API 生成 YAML 配置"""
    
    client = OpenAI(api_key=api_key)
    
    prompt = f"""你是一个机器学习架构专家。请根据以下需求，生成一个 YAML 格式的模型设计配置文件。

【需求描述】
{requirement}

【YAML 模板要求】
必须包含以下字段：
- constraints: 约束条件（max_params, input_shape, input_channels, num_classes, device）
- scoring: 评分方法（method, batch_size）
- proposal: 提案策略（style, temperature）
- search: 搜索参数（max_rounds, candidates_per_round）

【输出要求】
1. 只输出 YAML 内容，不要其他解释
2. 数值要合理（如 max_params 根据设备类型设置）
3. input_shape 格式为 [C, H, W] 或 [seq_len]
4. device 根据需求设置为 "cpu" 或 "cuda"

请生成 YAML："""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是专业的模型架构配置生成器，只输出有效的 YAML。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,  # 低温度，更确定
        max_tokens=1000
    )
    
    yaml_content = response.choices[0].message.content.strip()
    
    # 清理可能的 markdown 标记
    if yaml_content.startswith("```yaml"):
        yaml_content = yaml_content[7:]
    if yaml_content.endswith("```"):
        yaml_content = yaml_content[:-3]
    
    return yaml_content.strip()


def main():
    parser = argparse.ArgumentParser(description="Generate model design config via Token API")
    parser.add_argument("--requirement", required=True, help="需求描述，如：设计一个10万参数以内的文本情感分类模型")
    parser.add_argument("--output", default=None, help="输出文件路径（默认打印到屏幕）")
    parser.add_argument("--model", default="gpt-4", help="使用的模型名称")
    args = parser.parse_args()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("错误: 请设置 OPENAI_API_KEY 环境变量")
        sys.exit(1)
    
    print(f"🤖 正在使用 {args.model} 生成配置...")
    yaml_content = generate_config_with_api(args.requirement, api_key, args.model)
    
    if args.output:
        # 保存到文件
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        print(f"✅ 配置已保存到: {args.output}")
        
        # 验证 YAML 格式
        try:
            import yaml
            with open(args.output, "r") as f:
                config = yaml.safe_load(f)
            print(f"✅ YAML 验证通过，包含字段: {list(config.keys())}")
        except Exception as e:
            print(f"⚠️  YAML 验证失败: {e}")
    else:
        # 打印到屏幕
        print("\n" + "="*50)
        print(yaml_content)
        print("="*50)


if __name__ == "__main__":
    main()
