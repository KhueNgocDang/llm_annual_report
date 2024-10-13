import json

import numpy as np
import pandas as pd
import tiktoken
from mistletoe import Document
from mistletoe.ast_renderer import AstRenderer
from openai import OpenAI

PROMPT_TEMPLATAE: str = """
You are a helpful assistant designed to validate the content of sections in an annual report. The content to validate is related to climate change. You will be given a single row of Vietnamese text, and your task is to determine whether the data matches the provided criteria.

- Translate the text to English.
- You will treat natural disasters as climate change. If there is natural disaster risk and preventive action for that risk, it's the same as climate change.
- You can be lax on the criteria as not need to be too strict on whether it meet half of the requirement

**Return only a JSON object** with the following two properties:

- `"is_valid"`: a boolean (`true` or `false`) indicating whether the text matches the criteria.
- `"reason"`: Provide a brief explanation on while the text data is valid or not.

Both JSON properties must always be present.

Do not include any additional text or explanations outside the JSON object.

TEXT DATA:
```{content}```
CRITERIA
```{criteria}```
"""


def generate_category_embeddings(
    openai_client: OpenAI, embedding_model: str
) -> pd.DataFrame:
    """
    Generate embeddings for the categories
    Parameters
    ----------
        embedding_model
            model to use for embedding
    Returns
    -------
        pd.DataFrame
            dataframe containing the category embeddings
    """
    category_definition_df: pd.DataFrame = pd.DataFrame(
        [
            {
                "category": "CC1",
                "description": "đánh giá các rủi ro (các quy định, tác động vật lý hoặc các tác động chung) liên quan đến biến đổi khí hậu và các hành động đã hoặc sẽ thực hiện để quản lý rủi ro.",
            },
            {
                "category": "CC2",
                "description": "đánh giá các tác động tài chính hiện tại (và tương lai), tác động kinh doanh và cơ hội của biến đổi khí hậu.",
            },
            {
                "category": "GHG1",
                "description": "Mô tả các phương pháp sử dụng để tính toán khí thải nhà kính.",
            },
            {
                "category": "GHG2",
                "description": "Có sự xác nhận bởi yếu tố bên ngoài về lượng phát thải khí nhà kính hay không?– nếu có bởi ai và trên cơ sở gì",
            },
            {
                "category": "GHG3",
                "description": "Lượng phát thải khí nhà kính tính bằng đơn vị MtCO2e (hệ mét tấn CO2 thải ra).",
            },
            {
                "category": "GHG4",
                "description": "Việc công bố liên quan đến Phạm vi 1, Phạm vi 2, Phạm vi 3 và liên quan trực tiếp đến phát thải khí nhà kính. ",
            },
            {
                "category": "GHG5",
                "description": "Công bố phát thải nhà kính dựa trên nguồn phát thải nào?",
            },
            {
                "category": "GHG6",
                "description": "Công bố phát thải khí nhà kính dựa trên cơ sở vật chất hoặc cấp độ nào?",
            },
            {
                "category": "GHG7",
                "description": "So sánh lượng phát thải khí nhà kính trong năm hiện tại và năm trước",
            },
            {"category": "EC1", "description": "Lượng năng lượng tiêu thụ."},
            {
                "category": "EC2",
                "description": "Lượng năng lượng tiêu thụ có nguồn gốc từ nguồn năng lượng tái tạo.",
            },
            {
                "category": "EC3",
                "description": "Tiết lộ dựa trên loại khí thải, cơ sở vật chất hoặc cấp độ nào?",
            },
            {
                "category": "RC1",
                "description": "Giải thích chi tiết về chiến lược và kế hoạch giảm thiểu phát thải khí nhà kính.",
            },
            {
                "category": "RC2",
                "description": "Mục tiêu cụ thể về lượng giảm thiểu phát thải nhà kính",
            },
            {
                "category": "RC3",
                "description": "Lượng giảm phát thải tối đa và các chi phí hoặc tiết kiệm liên quan đến giảm phát khí thải nhà kính tính đến thời điểm báo cáo.",
            },
            {
                "category": "RC4",
                "description": "Mức độ chi phí liên quan đến phát thải khí nhà kính trong tương lai vì chi phí này được bao gồm trong kế hoạch sử dụng vốn của công ty.",
            },
            {
                "category": "ACC1",
                "description": "Giải thích ai là Ủy ban hoặc Giám đốc chịu trách nhiệm về các chính sách liên quan đến biến đổi khí hậu.",
            },
            {
                "category": "ACC2",
                "description": "Giải thích cơ chế xem xét khi đạt được mục tiêu công ty liên quan đến biến đổi khí hậu bởi Ủy ban hoặc Hội đồng quản trị",
            },
        ]
    )
    category_definition_df["embedding"] = get_embeddings(
        list_of_text=category_definition_df["description"].values,
        embedding_model=embedding_model,
        openai_client=openai_client,
    )
    return category_definition_df


def get_content(block: dict) -> str | None:
    """
    Recursive function to extract content from a block of mistletoe AST
    Parameters
    ----------
        block
            mistletoe AST block
    Returns
    -------
        str | None
            content of the block or its children
    """
    if "content" in block:
        return block["content"]
    elif "children" in block:
        if block["type"] == "Table":
            return "\n".join([get_content(row) for row in block["children"]])
        elif block["type"] == "TableRow":
            return "\t".join([get_content(cell) for cell in block["children"]])
        return "\n".join([get_content(child) for child in block["children"]])
    else:
        return None


def generate_ast_from_markdown(markdown_file_path: str) -> dict:
    """
    Generate AST from markdown text
    Parameters
    ----------
        markdown_file_path
            path to the markdown file
    Returns
    -------
        dict
            AST of the markdown text
    """
    with open(markdown_file_path, "r") as file:
        with AstRenderer() as renderer:
            doc: Document = Document(file.read())
            return json.loads(renderer.render(doc))


def generate_chunks_from_ast(
    openai_client: OpenAI, ast: dict, embedding_model: str
) -> pd.DataFrame:
    """
    Generate chunks from AST
    Parameters
    ----------
        ast
            AST of the markdown text
        embedding_model
            model to use for embedding
    Returns
    -------
        pd.DataFrame
            dataframe containing the chunks, their embedding, and their metadata
    """
    encoding: tiktoken.Encoding = tiktoken.encoding_for_model(embedding_model)
    chunks: list[dict] = []
    chunk_content: str = ""
    chunk_count: int = 0
    chunk_start_line: int
    chunk_token_count: int
    for block in ast["children"]:
        if chunk_content == "":
            chunk_start_line = block["line_number"]
        chunk_content += get_content(block) + "\n"
        chunk_token_count = len(encoding.encode(chunk_content))
        if chunk_token_count > 1024:
            chunk_count += 1
            chunks.append(
                {
                    "start_line": chunk_start_line,
                    "content": chunk_content,
                    "chunk_no": chunk_count,
                }
            )
            chunk_content = ""
    chunks_df: pd.DataFrame = pd.DataFrame(chunks)
    chunks_df["embedding"] = get_embeddings(
        list_of_text=chunks_df["content"].values,
        embedding_model=embedding_model,
        openai_client=openai_client,
    )
    return chunks_df


def cosine_similarity(a, b):
    """
    Perform similiary between two vectors using cosine similarity
    Parameters
    ----------
        a
            first vector
        b
            second vector
    Returns
    -------
        float
            cosine similarity between the two vectors
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_embeddings(
    openai_client: OpenAI,
    list_of_text: list[str],
    embedding_model: str,
    **kwargs,
) -> list[float]:
    """
    Get embeddings for a list of text using OpenAI API
    Parameters
    ----------
        openai_client
            OpenAI API client
        list_of_text
            list of text to embed
        embedding_model
            model to use for embedding
        **kwargs
            additional arguments to pass to the API
    Returns
    -------
        list
            list of embeddings of the texts
    """
    assert (
        len(list_of_text) <= 2048
    ), "The batch size should not be larger than 2048."

    # replace newlines, which can negatively affect performance.
    list_of_text: list[str] = [
        text.replace("\n", " ") for text in list_of_text
    ]

    data: dict[str, any] = openai_client.embeddings.create(
        input=list_of_text, model=embedding_model, **kwargs
    ).data
    return [d.embedding for d in data]


def similarity_search_on_cat(
    data_df: pd.DataFrame,
    cat_df: pd.DataFrame,
    cat_id: str,
    n: int,
    get_border: bool = False,
) -> pd.DataFrame:
    """
    Perform similarity search on a category
    Parameters
    ----------
        data_df
            dataframe containing the text data and its metadata
        cat_df
            dataframe containing the category embeddings
        cat_id
            category id
        n
            number of similar rows to return
        get_border
            whether to include the bordering chunks
    Returns
    -------
        pd.DataFrame
            dataframe containing the similar rows
    """
    df: pd.DataFrame = data_df.copy()
    embedding: list[float] = cat_df[cat_df["category"] == cat_id][
        "embedding"
    ].values[0]
    df["similarities"] = df["embedding"].apply(
        lambda x: cosine_similarity(x, embedding)
    )
    semantic_search_res: pd.DataFrame = (
        df.sort_values("similarities", ascending=False)
        .head(n)
        .sort_values("start_line")
    )
    chunk_list: list[int] = semantic_search_res["chunk_no"].to_list()
    if get_border:
        min_chunk: int = df["chunk_no"].min()
        max_chunk: int = df["chunk_no"].max()
        before_chunk: int
        after_chunk: int
        for chunk in semantic_search_res["chunk_no"].to_list():
            before_chunk = chunk - 1
            if before_chunk >= min_chunk:
                chunk_list.append(before_chunk)
            after_chunk = chunk + 1
            if after_chunk <= max_chunk:
                chunk_list.append(after_chunk)
    return df[df["chunk_no"].isin(set(chunk_list))].sort_values("start_line")


def generate_prompt(
    prompt_template: str,
    data_df: pd.DataFrame,
    cat_df: pd.DataFrame,
    cat_id: str,
    n: int,
) -> tuple[str, list[dict]]:
    """
    Generate a prompt for the semantic search
    Parameters
    ----------
        prompt_template
            template for the prompt
        data_df
            dataframe containing the text data and its metadata
        cat_df
            dataframe containing the category embeddings
        cat_id
            category id
        n
            number of similar rows to return
    Returns
    -------
        tuple
            prompt and the similar rows' metadata
    """
    sematic_search_results: pd.DataFrame = similarity_search_on_cat(
        data_df=data_df, cat_df=cat_df, cat_id=cat_id, n=n
    )
    prompt: str = prompt_template.format(
        content="\n".join(sematic_search_results["content"].values),
        criteria=cat_df[cat_df["category"] == cat_id]["description"].values[0],
    )
    return prompt, sematic_search_results[["content", "start_line"]].to_dict(
        orient="records"
    )


def scoring_sematic_search_results(
    openai_client: OpenAI,
    prompt_template: str,
    document_df: pd.DataFrame,
    category_df: pd.DataFrame,
    category_id: str,
    n: int,
    reasoning_model: str,
) -> dict[str, any]:
    """
    Score the semantic search results
    Parameters
    ----------
        openai_client
            OpenAI API client
        prompt_template
            template for the prompt
        document_df
            dataframe containing the text data and its metadata
        category_df
            dataframe containing the category embeddings
        category_id
            category id
        n
            number of similar rows to return
        reasoning_model
            model to use for reasoning
    Returns
    -------
        dict
            result of the scoring
    """
    prompt: str
    sematic_search_result: list[dict]
    prompt, sematic_search_result = generate_prompt(
        prompt_template=prompt_template,
        data_df=document_df,
        cat_df=category_df,
        cat_id=category_id,
        n=n,
    )
    response: dict[str, any] = openai_client.chat.completions.create(
        model=reasoning_model,
        messages=[{"role": "user", "content": prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "criteria_response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "is_valid": {
                            "description": "a boolean (`true` or `false`) indicating whether the text matches the criteria.",
                            "type": "string",
                        },
                        "reason": {
                            "description": """Provide a brief explanation on while the text data is valid or not""",
                            "type": "string",
                        },
                        "additionalProperties": False,
                    },
                },
            },
        },
    )
    result: dict[str, any] = json.loads(response.choices[0].message.content)
    return {
        "is_valid": result["is_valid"],
        "reason": result["reason"],
        "category_id": category_id,
        "prompt": prompt,
        "metadata": sematic_search_result,
    }


def scoring_markdown_annual_report(
    markdown_file_path: str,
    embedding_model: str,
    openai_client: OpenAI,
    reasoning_model: str,
    n: int,
) -> pd.DataFrame:
    """
    Score the markdown annual report
    Parameters
    ----------
        markdown_file_path
            path to the markdown file
        embedding_model
            model to use for embedding
        openai_client
            OpenAI API client
        reasoning_model
            model to use for reasoning
        n
            number of similar rows to return
    Returns
    -------
        pd.DataFrame
            result of the scoring
    """
    chunked_document_df: pd.DataFrame = generate_chunks_from_ast(
        openai_client=openai_client,
        ast=generate_ast_from_markdown(markdown_file_path=markdown_file_path),
        embedding_model=embedding_model,
    )
    category_embeddings_df: pd.DataFrame = generate_category_embeddings(
        openai_client=openai_client, embedding_model=embedding_model
    )
    scoring: list[dict[str, any]] = []
    for category_id in category_embeddings_df["category"].to_list():
        print(f"Scoring category {category_id}")
        scoring.append(
            scoring_sematic_search_results(
                openai_client=openai_client,
                document_df=chunked_document_df,
                category_df=category_embeddings_df,
                category_id=category_id,
                prompt_template=PROMPT_TEMPLATAE,
                reasoning_model=reasoning_model,
                n=n,
            )
        )
    return pd.DataFrame(scoring)
