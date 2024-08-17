import os
from langchain.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import CharacterTextSplitter
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
import gradio as gr
import base64
from IPython.display import HTML, display

nvapi_key = "nvapi-...(your NVIDIA NIM API_KEY)"
os.environ["NVIDIA_API_KEY"] = nvapi_key


def process_text(file, user_prompt):
    # 处理txt文本输入
    # 指定LLM模型
    llm = ChatNVIDIA(model="microsoft/phi-3-small-128k-instruct", nvidia_api_key=nvapi_key, max_tokens=512)
    result = llm.invoke(user_prompt)
    html = '<ul>'
    for doc in result:
        html += f'<li>{doc}</li>'
    html += '</ul>'
    # 指定文本向量化模型
    embedder = NVIDIAEmbeddings(model="ai-embed-qa-4")
    # 读取数据文件
    data = []
    sources = []
    if file.endswith('.txt'):
        with open(file, encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                if len(line) >= 1:
                    data.append(line)
                    sources.append(file)
    # 进行一些基本的清理并删除空行
    documents = [d for d in data if d != '']
    # 文本存为本地向量数据库
    text_splitter = CharacterTextSplitter(chunk_size=400, separator=" ")
    docs = []
    metadatas = []

    for i, d in enumerate(documents):
        splits = text_splitter.split_text(d)
        docs.extend(splits)
        metadatas.extend([{"source": sources[i]}] * len(splits))
    store = FAISS.from_texts(docs, embedder, metadatas=metadatas)
    store.save_local('./zh_data/nv_embedding')
    # 读取向量数据库
    store = FAISS.load_local("./zh_data/nv_embedding", embedder, allow_dangerous_deserialization=True)
    # 提出问题并基于phi-3-small-128k-instruct模型进行RAG检索
    retriever = store.as_retriever()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer solely based on the following context:\n<Documents>\n{context}\n</Documents>",
            ),
            ("user", "{question}"),
        ]
    )
    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    text = chain.invoke(user_prompt)
    return text, html


def image2b64(image_file):
    with open(image_file, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()
        return image_b64


def display_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    html = f'<img src="data:image/png;base64,{encoded_string}" />'
    return html


def process_image(file, user_prompt):
    # 处理图片输入
    # 将图片进行编解码
    image_b64 = image2b64(file)
    # 将编码后的图像按照格式给到Microsoft Phi 3 vision, 利用其强大能力解析图片中的数据
    chart_reading = ChatNVIDIA(model="ai-phi-3-vision-128k-instruct")
    # 调用invoke方法并传入提示词
    result = chart_reading.invoke(f'{user_prompt}: <img src="data:image/png;base64,{image_b64}" />')
    return result.content


def big_model_output(file, user_prompt):
    # 获取文件扩展名
    file_extension = os.path.splitext(file)[1].lower()

    if file_extension == ".txt":
        return process_text(file, user_prompt)
    elif file_extension in [".png", ".jpg", ".jpeg"]:
        image_html = display_image(file)  # 获取图像的HTML标签字符串
        return process_image(file, user_prompt), image_html
    else:
        return "Invalid input type"


iface = gr.Interface(
    fn=big_model_output,
    inputs=[
        gr.File(),
        gr.Textbox(lines=1)
    ],
    outputs=[
        "text",  # 文本输出
        gr.HTML()  # 图像输出
    ],
    title="多模态RAG对话AI智能体",
    description="输入文字或图像，大模型会进行分析输出。"
)

iface.launch()
