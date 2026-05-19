import streamlit as st
import numpy as np
import pandas as pd
import random
import os
import shutil
import h5py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D
import pickle

# ==========================================
# 🔥 核心黑科技：Windows转Linux路径拦截清洗器
# ==========================================
def fix_windows_h5_bug(original_file):
    fixed_file = "linux_fixed_" + original_file
    if not os.path.exists(fixed_file):
        shutil.copy(original_file, fixed_file)
        with h5py.File(fixed_file, 'r+') as f:
            keys = list(f.keys())
            for k in keys:
                if '\\' in k:
                    new_k = k.replace('\\', '/')
                    f.move(k, new_k)
    return fixed_file

# ==========================================
# 1. 纯净模型加载 
# ==========================================
@st.cache_resource
def load_surrogate_model():
    model = Sequential()
    model.add(Conv1D(activation="relu", input_shape=(8, 4), padding='same', filters=256, kernel_size=3))
    model.add(Conv1D(activation="relu", padding='same', filters=256, kernel_size=3))
    model.add(Dropout(0.3))
    model.add(Conv1D(activation="relu", padding='same', filters=256, kernel_size=3))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # 洗掉 Windows 带来的反斜杠 Bug
    fixed_weight_path = fix_windows_h5_bug('shap_model.weights.h5')
    model.load_weights(fixed_weight_path)
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
        
    return model, scaler

def seq_to_onehot(seq):
    nuc_d = {'a': [1, 0, 0, 0], 'c': [0, 1, 0, 0], 'g': [0, 0, 1, 0], 't': [0, 0, 0, 1]}
    vector = np.array([nuc_d.get(base, [0, 0, 0, 0]) for base in seq.lower()])
    return vector.reshape(1, len(seq), 4)

# ==========================================
# 2. 核心算法：逆向序列生成引擎 (完美阶梯画图版)
# ==========================================
def reverse_engineer_sequence(target_value, model, scaler, seq_len=8, iterations=500):
    bases = ['a', 'c', 'g', 't']
    current_seq = "".join(random.choices(bases, k=seq_len))

    def get_pred(seq):
        x = seq_to_onehot(seq)
        scaled_pred = model.predict(x, verbose=0)[0][0]
        if scaler:
            return scaler.inverse_transform([[scaled_pred]])[0][0]
        return scaled_pred

    current_pred = get_pred(current_seq)
    current_loss = abs(current_pred - target_value)
    
    # 记录初始状态 (第 0 代)
    history = [{'iteration': 0, 'sequence': current_seq.upper(), 'predicted_value': current_pred, 'loss': current_loss}]
    progress_bar = st.progress(0)

    for i in range(iterations):
        mut_pos = random.randint(0, seq_len - 1)
        new_base = random.choice([b for b in bases if b != current_seq[mut_pos]])
        new_seq = current_seq[:mut_pos] + new_base + current_seq[mut_pos + 1:]
        
        new_pred = get_pred(new_seq)
        new_loss = abs(new_pred - target_value)

        # 适者生存：如果突变产生的分数更接近目标，就更新当前最佳血统
        if new_loss < current_loss:
            current_seq = new_seq
            current_pred = new_pred
            current_loss = new_loss

        # 🌟 核心修正：无论刚才的突变是成功还是失败，都把【当前这一代存活下来的最优战绩】记录下来！
        # 这样才能画出中间平稳、偶尔下降的漂亮阶梯图
        history.append({'iteration': i + 1, 'sequence': current_seq.upper(), 'predicted_value': current_pred, 'loss': current_loss})
        
        # 如果达到了极度精确的偏差，提前结束寻优
        if current_loss < 0.01:
            break
            
        progress_bar.progress((i + 1) / iterations)
        
    return current_seq.upper(), current_pred, pd.DataFrame(history)

# ==========================================
# 3. 网页 UI 设计 (Streamlit)
# ==========================================
st.set_page_config(page_title="5'UTR 理性设计平台", layout="centered")
st.title("🧬 5'UTR 表达强度逆向设计系统")
st.markdown("基于 CNN 代理模型与定向进化算法，输入预期的蛋白表达分数，智能生成对应的核苷酸序列。模型会自动遵循底层序列语法。")

model, scaler = load_surrogate_model()
st.divider()

col1, col2 = st.columns(2)
with col1:
    target_expr = st.number_input("🎯 设定预期的表达强度 (Target Value)", value=150000.0, step=1000.0)
with col2:
    iterations = st.slider("🔄 进化迭代次数 (Iterations)", min_value=100, max_value=2000, value=500, step=100)

generate_btn = st.button("🚀 开始逆向生成序列", use_container_width=True)

if generate_btn:
    with st.spinner('算法正在广阔的序列空间中搜寻最优解...请稍候'):
        final_seq, final_pred, history_df = reverse_engineer_sequence(target_expr, model, scaler, seq_len=8, iterations=iterations)
        
        st.success("序列生成完毕！")
        st.subheader("🏆 最优生成结果")
        st.metric(label="生成的核苷酸序列", value=final_seq)
        
        c1, c2 = st.columns(2)
        c1.metric(label="CNN 预测表达值", value=f"{final_pred:.4f}", delta=f"偏离目标: {final_pred - target_expr:.4f}", delta_color="inverse")
        c2.metric(label="目标表达值", value=f"{target_expr:.4f}")
        
        st.subheader("📈 序列进化轨迹")
        # 强制绘制以 iteration 为横坐标的折线图
        st.line_chart(history_df.set_index('iteration')['predicted_value'])
        
        with st.expander("查看详细突变历史"):
            st.dataframe(history_df)
