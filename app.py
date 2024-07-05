#!/usr/bin/python
# -*- coding: utf-8 -*-
import streamlit as st
import logging
import torch
import uuid
from transformers import AutoModelForCausalLM, AutoTokenizer


base_path = './RAG_models'

# 检查目标目录是否存在，如果不存在则克隆仓库
if not os.path.exists(base_path):
    clone_command = f'git clone https://ent-app-dev:1ca51f9b37ac0c2ecfdaeb509718ec5ca39835c3@code.openxlab.org.cn/ent-app-dev/RAG_models.git {base_path}'
    clone_result = os.system(clone_command)
    if clone_result != 0:
        raise RuntimeError(f"Failed to clone repository with command: {clone_command}")

    # 安装 Git LFS
    lfs_install_command = 'git lfs install'
    print(f"Running LFS install command: {lfs_install_command}")
    lfs_install_result = os.system(lfs_install_command)
    if lfs_install_result != 0:
        raise RuntimeError(f"Failed to install Git LFS with command: {lfs_install_command}")

    # 拉取 LFS 文件
    lfs_pull_command = f'cd {base_path}/Sft_model && git lfs pull'

    try:
        lfs_pull_result = subprocess.run(lfs_pull_command, shell=True, check=True, capture_output=True)
        print(f"LFS pull output for: {lfs_pull_result.stdout.decode()}")
    except subprocess.CalledProcessError as e:
        print(f"LFS pull error output: {e.stderr.decode()}")
        raise RuntimeError(f"Failed to pull LFS files with command: {lfs_pull_command}")


# 目标GitHub仓库的URL
GITHUB_REPO_URL = "https://ent-app-dev:1ca51f9b37ac0c2ecfdaeb509718ec5ca39835c3@code.openxlab.org.cn/ent-app-dev/SFT_log.git"

# 检查当前目录下是否存在./log文件夹，如果不存在则创建它
log_dir = './log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    print(f"Created directory: {log_dir}")
else:
    print(f"Directory already exists: {log_dir}")

# 初始化Git仓库（如果尚未初始化）
if not os.path.exists('.git'):
    subprocess.run(['git', 'init'], check=True)
    subprocess.run(['git', 'remote', 'add', 'origin', GITHUB_REPO_URL], check=True)

# 添加./log文件夹到Git
subprocess.run(['git', 'add', './log'], check=True)

# 提交更改
commit_message = "Add or update ./log folder"
subprocess.run(['git', 'commit', '-m', commit_message], check=True)

# 推送更改到GitHub仓库
subprocess.run(['git', 'push', '-u', 'origin', 'main'], check=True)





# 创建日志配置函数
def setup_logger():
    if 'logger_configured' not in st.session_state:
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(log_format)

        # 创建一个文件处理器，并赋予一个唯一的名称
        file_handler = logging.FileHandler('./log/st_log.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        file_handler_name = 'streamlit_file_handler'
        streamlit_root_logger = logging.getLogger(st.__name__)

        # 检查处理器是否已经存在
        if not any(handler.get_name() == file_handler_name for handler in streamlit_root_logger.handlers):
            file_handler.set_name(file_handler_name)
            streamlit_root_logger.addHandler(file_handler)

        st.session_state['logger_configured'] = True

# 在应用启动时配置日志
setup_logger()

# 使用日志记录器
streamlit_root_logger = logging.getLogger(st.__name__)

# 生成或获取用户 ID
if 'user_id' not in st.session_state:
    st.session_state['user_id'] = str(uuid.uuid4())

user_id = st.session_state['user_id']
streamlit_root_logger.info(f"New session started with user ID: {user_id}")

# 设置模型路径并加载模型和tokenizer
if 'model' not in st.session_state:
    base_path = './RAG_models'
    tokenizer = AutoTokenizer.from_pretrained(base_path + '/Sft_model', trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base_path + '/Sft_model', trust_remote_code=True, torch_dtype=torch.float16).cuda()
    st.session_state['model'] = model
    st.session_state['tokenizer'] = tokenizer
else:
    model = st.session_state['model']
    tokenizer = st.session_state['tokenizer']

def chat(message, history):
    streamlit_root_logger.info(f"User ({user_id}) query: {message}")
    for response, history in model.stream_chat(tokenizer, message, history, top_p=0.7, temperature=1):
        streamlit_root_logger.info(f"Response to {user_id}: {response}")
        yield response

# def send_query(query):
#     query = f'<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n'
#     input_ids = tokenizer.encode(query, return_tensors="pt").to("cuda")
#     output_ids = model.generate(
#         input_ids,
#         max_length=1024,  # 设置生成文本的最大长度
#         temperature=1,
#         top_k=50,
#         top_p=1.0,
#     )
#     output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
#     return output_text

# 创建 Streamlit UI 组件
st.title("商飞大模型测试")
user_input = st.text_input("请输入你的关于飞机设计相关问题", placeholder="请在这里输入（中英文均支持）...")

if 'result' not in st.session_state:
    st.session_state['result'] = None

if 'feedback_given' not in st.session_state:
    st.session_state['feedback_given'] = False

if st.button("开始回答"):
    if user_input:
        # Display user's query
        st.write(f"User ({user_id}): {user_input}")
        streamlit_root_logger.info(f"User ({user_id}) query (Extract keywords): {user_input}")

        # Get and display the response from the chat server
        try:
            history = []
            responses = []
            for response in chat(user_input, history):
                responses.append(response)
            result = responses[-1]
            st.write(f"Bot: {result}")
            st.session_state['result'] = result
            st.session_state['feedback_given'] = False
            streamlit_root_logger.info(f"Response (Extract keywords): {result}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
            streamlit_root_logger.error(f"Error (Extract keywords): {e}")
    else:
        st.error("Please enter a query to send.")

# 加入用户评价功能
if st.session_state['result'] is not None and not st.session_state['feedback_given']:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("满意"):
            streamlit_root_logger.info(f"User satisfaction (满意) for user ({user_id})")
            st.session_state['feedback_given'] = True
            st.rerun()
    with col2:
        if st.button("不满意"):
            streamlit_root_logger.info(f"User satisfaction (不满意) for user ({user_id})")
            st.session_state['feedback_given'] = True
            st.rerun()

if st.session_state['feedback_given']:
    st.write("Thank you for your feedback!")
    st.session_state['result'] = None
