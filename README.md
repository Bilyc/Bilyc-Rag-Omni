​100行代码实现私人定制多模态RAG机器人
项目名称：私人定制多模态RAG机器人
报告日期：2024年8月18日

项目负责人：Bilyc（WsWs）

一、项目概述
该项目旨在构建多模态智能体对话机器人，以txt 文本为RAG知识库并结合图片识别功能。选用 Nvidia NIM 平台的相关模型，进行数据构建和向量化处理。通过环境搭建和代码实现，具备一定的可操作性。应用场景广泛，包括客户服务、教育培训、金融服务等多个领域。其亮点在于采用先进模型，能提供全新互动体验，且数据处理和功能整合较为有效。

二、技术方案与实施步骤
（一）LLM大模型的选择
模型选择： 选择微软的phi-3-small-128k-instruct和ai-phi-3-vision-128k-instruct模型，分别实现txt文档和png、jpg、jpeg图像的智能回答和识别。

（二）RAG模型的优势分析
Phi-3-Small-128k-Instruct 模型占用资源相对较少，便于部署和应用在资源有限的环境中，同时仍能提供较为准确和有用的指令响应。它在处理常见任务时能够保持高效和可靠。 AI-Phi-3-Vision-128k-Instruct 模型则在视觉相关的指令处理上表现出色。能够理解和处理图像相关的指令，为涉及图像分析、识别和理解的任务提供有力支持。这两个模型都具有长上下文处理能力，能更好地理解复杂的任务和指令，为用户提供更全面和准确的服务。

（三）数据的构建
1.数据构建过程
首先，收集txt电子书和图片。当选择txt电子书后，Phi-3-Small-128k-Instruct 模型会对文本进行预处理，建立本地Faiss vector向量数据库。而图片则直接喂给模型分析。

2.向量化处理方法及其优势。
向量化模型选用ai-embed-qa-4。对 txt 文档进行向量化。该模型会对文档中的文本进行深度分析和编码。它会将文本分解为单词、短语或更细粒度的语言单元，并通过其内部的神经网络架构将这些单元转换为数值向量。

（四）功能整合
采用gradio框架和Nvidia的NIM服务，将两个模型整合成一个多模态的智能机器人，并实现网页可视化应用。

三、实施步骤
（一）环境搭建
1. 创建Python环境
首先需要安装Miniconda：

大家可以根据自己的网络情况从下面的地址下载：

miniconda官网地址：https://docs.conda.io/en/latest/miniconda.html

清华大学镜像地址： Index of /anaconda/miniconda/ | 清华大学开源软件镜像站 | Tsinghua Open Source Mirror

安装完之后，打开Anaconda Powershell:

在打开的终端中按照下面的步骤执行,配置环境:

创建python 3.8虚拟环境

conda create --name ai_endpoint python=3.8.1
进入虚拟环境

conda activate ai_endpoint
安装nvidia_ai_endpoint工具

pip install langchain-nvidia-ai-endpoints
安装Jupyter Lab

pip install jupyterlab
安装langchain_core

pip install langchain_core
安装langchain和langchain_community

pip install langchain
pip install –U langchain_community
安装matplotlib

pip install matplotlib
安装Numpy

pip install numpy
安装faiss

pip install faiss-cpu==1.7.2
安装OPENAI库

pip install openai
2.Macbook 环境
Macbook也可以按照上面的步骤同样执行, 只是在下载Miniconda的时候要下载Mac OS版本的

（二）代码实现
直接上干货仅，有100行左右。

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

nvapi_key = "nvapi-（请在Nvdia NIM平台申请你自己的API_KEY）"
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
四、项目成果与展示：
（一）应用场景
客户服务与支持、教育培训领域、金融服务行业、医疗健康领域、企业内部应用、政务服务领域、媒体与新闻行业、旅游与出行领域等。

（二）功能演示
1.首先看图片的识别效果。


2.再看一看RAG文本的实现效果


五、项目总结与展望
（一）项目总结
本文旨在构建一个以txt文本为知识库和图片识别相结合的多模态智能体对话机器人，为读者提供全新互动体验。 选择Nvidia NIM平台的（phi-3-small-128k-instruct 和 ai-phi-3-vision-128k-instruct）模型，并采用了有效的数据构建和向量化处理方法。 详细介绍了环境搭建和代码实现，具有可操作性。总体而言，该项目具有一定的创新性和实用性，但在性能评估和未来规划方面还有待进一步完善。读者可在此基础上进行改进完善。

（二）未来方向
该项目未来可朝着以下方向发展：一是优化模型性能，提高对文本和图像的理解与分析能力，提升回答的准确性和全面性；二是拓展应用场景，深入挖掘更多领域的需求，为用户提供更广泛的服务；三是加强与其他技术的融合，如与物联网、大数据等结合，实现更智能的交互；四是持续改进用户体验，根据用户反馈不断优化界面和交互方式；五是探索商业化应用，为企业和个人带来实际价值。

附件与参考资料
1. https://python.langchain.com/v0.1/docs/integrations/chat/nvidia_ai_endpoints/

2. NVIDIA NIM页面： https://build.nvidia.com/explore/discover

3. NVIDIA DLI课程学习资料页面：https://www.nvidia.cn/training/online/

4. https://github.com/kinfey/Microsoft-Phi-3-NvidiaNIMWorkshop

​
