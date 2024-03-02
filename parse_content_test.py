import re
import csv

def generate_data(content, summary):
    # 用空格替换换行符
    content = content.replace('\n', ' ')
    summary = summary.replace('\n', ' ')

    # 生成 20 个不同的问题
    questions = [
        f"{content}象征着什么？",
        f"在易经中，{content}的含义是什么？",
        f"能解释{content}在{summary}方面的重要性吗？",
        f"{content}如何代表发展的初期阶段？",
        f"水与{content}有什么关系？",
        f"解释{content}关于山和水的象征意义。",
        f"为什么{content}在{content}解释中被认为是至关重要的？",
        f"KAN（水）和GEN（山）的顺序在{content}中代表什么？",
        f"{content}如何提示从混沌状态逐渐启蒙？",
        f"{content}为什么与培养美德的概念相关？",
        f"描述{content}中提到的混乱的初期阶段。",
        f"{content}解释中{content}在克服初期挑战中的作用是什么？",
        f"{content}如何与童稚无邪的概念相关？",
        f"解释《序卦》中的短语：“物生必{content}”。",
        f"“山下出泉，{content}”中的表述在{content}解释中暗示了什么？",
        f"{content}如何象征从混乱中逐渐清晰的出现？",
        f"在{content}表示的混乱时期，聪明的人会采取什么行动？",
        f"{content}如何鼓励果断行动来培育美德？",
        f"如何将{content}应用于生活各个方面的发展早期阶段？",
        f"解释《象》中的短语：“君子以果行育德”在{content}背景下的象征意义。"
    ]

    # 生成数据
    data = []
    for question in questions:
        data.append({'Question': question, 'Content': content, 'Summary': summary})

    return data

# 示例输入
ai_message_content = 'content:"蒙卦"\nsummary:"蒙卦象征着教育启蒙的智慧，在周易卦象中由两个异卦相叠组成：下卦坎（水）和上卦艮（山）。坎代表水，艮代表山，山下有水泉涌出，意味着初期阶段的蒙昧状态，而泉水始流出山，必将渐汇成江河，象征着蒙昧逐渐启蒙。蒙卦提示我们，在事物发展的初期阶段，必然会有困难和蒙昧，因此教育是至关重要的，需要培养学生纯正无邪的品质，以解除蒙昧状态。蒙卦具有启蒙和通达的象征意义。\n\n蒙卦位于《屯》卦之后，《序卦》中指出：“物生必蒙，故受之以蒙。蒙者，蒙也，特之稚也。”这表明在事物发展的初期，都会有蒙昧未开的状态，对于幼稚的物体或人来说，就是童蒙的阶段。\n\n在《象》中，解释蒙卦为“山下出泉，蒙；君子以果行育德”，意味着蒙昧时期有如泉水涌出山下，君子应该通过果断行动来培育德行。"\n'

# 提取内容和摘要
content_match = re.search(r'content:"([^"]*)"', ai_message_content)
summary_match = re.search(r'summary:"([^"]*)"', ai_message_content)

# 检查是否找到匹配项
if content_match and summary_match:
    content = content_match.group(1)
    summary = summary_match.group(1)

    # 生成数据
    data = generate_data(content, summary)

    # 写入 CSV 文件
    with open('zhouyi_dataset.csv', 'w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ['Question', 'Content', 'Summary']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # 写入头部
        writer.writeheader()

        # 写入数据
        writer.writerows(data)

    print('数据成功写入 zhouyi_dataset.csv 文件。')
else:
    print('未找到匹配项。')