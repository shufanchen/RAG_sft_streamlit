#!/usr/bin/python
# -*- coding: utf-8 -*-
import streamlit as st
import logging
import torch
import uuid
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import subprocess
from datetime import datetime
import shutil
import threading
import time
# 设置 PyTorch 的 CUDA 内存分配策略
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
# 清空未使用的显存
torch.cuda.empty_cache()

# 打印当前显存使用情况
print(torch.cuda.memory_summary(device=None, abbreviated=False))
#模型地址
base_path = './RAG_models'
#每次使用时间限制
use_time_limit = 300
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




# 检查当前目录下是否存在./log文件夹，如果不存在则创建它
log_dir = './log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    print(f"Created directory: {log_dir}")
else:
    print(f"Directory already exists: {log_dir}")
# 确定本次启动的日志文件名称
if 'log_file_path' not in st.session_state:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    log_file = f"st_log_{timestamp}.log"
    log_file_path = os.path.join(log_dir, log_file)
    st.session_state['log_file_path'] = log_file_path

    # 创建本次启动的日志文件
    with open(log_file_path, 'w') as file:
        pass  # 这将创建一个空文件
else:
    log_file_path = st.session_state['log_file_path']



# 创建日志配置函数
def setup_logger():
    if 'logger_configured' not in st.session_state:
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(log_format)

        # 创建一个文件处理器，并赋予一个唯一的名称
        file_handler = logging.FileHandler(log_file_path)
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
while 'model' not in st.session_state:
    try:
        base_path = './RAG_models'
        tokenizer = AutoTokenizer.from_pretrained(base_path + '/Sft_model', trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(base_path + '/Sft_model', trust_remote_code=True, torch_dtype=torch.float16).cuda()
        st.session_state['model'] = model
        st.session_state['tokenizer'] = tokenizer
    except RuntimeError as e:
        if 'out of memory' in str(e):
            st.error("目前使用人数较多，显存不足，请稍后再试")
            torch.cuda.empty_cache()  # 清空未使用的显存
            time.sleep(use_time_limit)
        raise e  #抛出异常


# 加载成功后，使用模型和tokenizer
model = st.session_state['model']
tokenizer = st.session_state['tokenizer']



def chat(message, history):
    streamlit_root_logger.info(f"User ({user_id}) query: {message}")
    for response, history in model.stream_chat(tokenizer, message, history, top_p=0.7, temperature=1):
        #streamlit_root_logger.info(f"Response to {user_id}: {response}")
        yield response


# 创建 Streamlit UI 组件
st.title("商飞大模型测试")
user_input = st.text_input("请输入你的关于飞机设计相关问题(注意目前单次会话最长时间为5分钟)", placeholder="请在这里输入（中英文均支持）...")

if 'result' not in st.session_state:
    st.session_state['result'] = None

if 'feedback_given' not in st.session_state:
    st.session_state['feedback_given'] = False

if st.button("开始回答"):
    if user_input:
        # Display user's query
        st.write(f"User ({user_id}): {user_input}")
        streamlit_root_logger.info(f"User ({user_id}) query : {user_input}")

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
            streamlit_root_logger.info(f"Response to {user_id}: {result}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
            streamlit_root_logger.error(f"Error of {user_id}: {e}")
    else:
        st.error("Please enter a query to send.")

# 加入用户评价功能
if st.session_state['result'] is not None and not st.session_state['feedback_given']:
    col1, col2, col3 = st.columns(3)
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
    with col3:
        if st.button("建议"):
            st.session_state['give_suggestion'] = True

if st.session_state.get('give_suggestion', False):
    suggestion = st.text_area("请输入您的建议：")
    if st.button("提交建议"):
        if suggestion:
            streamlit_root_logger.info(f"User suggestion for user ({user_id}): {suggestion}")
            st.session_state['give_suggestion'] = False
            st.session_state['feedback_given'] = True
        else:
            st.error("建议不能为空，请输入您的建议。")
        st.rerun()

if st.session_state['feedback_given']:
    st.write("Thank you for your feedback!")
    st.write(f"Bot: {st.session_state['result']}")

def send_log_file():
    repo_url = "https://ent-app-dev:1ca51f9b37ac0c2ecfdaeb509718ec5ca39835c3@code.openxlab.org.cn/ent-app-dev/SFT_log.git"
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    repo_dir = f'./SFT_log_{timestamp}'
        
    # 克隆远程仓库
    try:
        subprocess.run(['git', 'clone', repo_url, repo_dir], check=True)
        print(f"Cloned repository to {repo_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to clone repository: {e}")

    # 复制日志文件到克隆的仓库目录
    shutil.copy2(log_file_path, repo_dir)
    print(f"Copied {log_file_path} to {repo_dir}")

    # 保存当前工作目录
    original_working_dir = os.getcwd()
    # 切换到克隆的仓库目录
    os.chdir(repo_dir)
    # 设置 Git 用户名和邮箱
    try:
        subprocess.run(['git', 'config', 'user.name', 'RAG_sft_test'], check=True)
        subprocess.run(['git', 'config', 'user.email', 'RAG_sft_test@pjlab.org.cn'], check=True)
        print("Configured Git user name and email.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to configure Git user name and email: {e}")

    # 添加日志文件到 Git
    subprocess.run(['git', 'add', '-A'], check=True)

    try:
        # 提交更改
        commit_message = f"Add log file {log_file}"
        subprocess.run(['git', 'commit', '-m', commit_message], check=True)

        # 推送更改到远程仓库
        subprocess.run(['git', 'push', '-u', 'origin', 'main'], check=True)
        print("Pushed log file to GitHub repository.")
    except Exception as e:
        print(f"An error occurred while running subprocess: {e}")
    os.chdir(original_working_dir)


def send_log_every_time():
     # 释放模型和显存
    time.sleep(use_time_limit)  
    if 'model' in st.session_state:
        del st.session_state['model']
    if 'tokenizer' in st.session_state:
        del st.session_state['tokenizer']
    torch.cuda.empty_cache()     
    send_log_file()


# 创建并启动后台线程
if 'log_thread_started' not in st.session_state:
    log_thread = threading.Thread(target=send_log_every_time, daemon=True)
    log_thread.start()
    st.session_state['log_thread_started'] = True
