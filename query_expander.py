# query_expander.py


EXPANSION_DICT = {

    # 论文分析
    "方法": [
        "method",
        "methodology",
        "approach",
        "framework",
        "architecture",
        "algorithm"
    ],

    "method": [
        "method",
        "methodology",
        "approach",
        "framework",
        "architecture",
        "algorithm"
    ],


    # 实验相关
    "实验": [
        "experiment",
        "evaluation",
        "setup",
        "benchmark",
        "result"
    ],

    "experiment": [
        "experiment",
        "evaluation",
        "setup",
        "benchmark",
        "result"
    ],


    # 数据集
    "数据集": [
        "dataset",
        "benchmark",
        "corpus",
        "data"
    ],


    # 结果
    "结果": [
        "result",
        "performance",
        "accuracy",
        "evaluation"
    ],


    # 模型
    "模型": [
        "model",
        "architecture",
        "network"
    ],

    "推理": [
        "reasoning",
        "rationale",
        "chain-of-thought",
        "CoT"
    ],

    "行动": [
        "action",
        "acting",
        "tool interaction"
    ],

    "结合": [
        "combine",
        "integrate",
        "interaction"
    ],

    "幻觉": [
        "hallucination",
        "factual accuracy"
    ]
}


def expand_query(query: str) -> str:

    expanded_terms = []

    query_lower = query.lower()


    for key, values in EXPANSION_DICT.items():

        if key in query_lower:

            expanded_terms.extend(values)


    if expanded_terms:

        unique_terms = list(dict.fromkeys(expanded_terms))

        expanded_query = (
            query
            + " "
            + " ".join(unique_terms)
        )

        return expanded_query


    return query
