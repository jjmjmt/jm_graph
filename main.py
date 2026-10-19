import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px

# 1. 브라우저 탭 설정 및 앱 제목 구성
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 영화 유형 나누기 앱")

# 2. 데이터 불러오기 (인코딩 UTF-8)
@st.cache_data
def load_data():
    url = "https://githubusercontent.com"
    df = pd.read_csv(url, encoding="utf-8")
    return df

try:
    df_raw = load_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# 3. 데이터 전처리 및 파생 속성 생성
df = df_raw.copy()

# 네 속성 중 값이 없거나 첫 주 관객이 0인 영화 필터링
df = df.dropna(subset=['first_scrn', 'total_audi', 'days_in_top10', 'first_week_audi'])
df = df[df['first_week_audi'] > 0]

# 상용로그 및 롱런 지수 생성
df['log_first_scrn'] = np.log10(df['first_scrn'])
df['log_total_audi'] = np.log10(df['total_audi'])
df['long_run_idx'] = (df['total_audi'] / df['first_week_audi']).clip(upper=20)

# 전체 편수와 묶은 편수를 화면에 한 줄로 출력
total_count = len(df_raw)
filtered_count = len(df)
st.info(f"📊 전체 편수: {total_count}편 | 묶은 편수: {filtered_count}편")

# 4. 속성 선택 UI (둘 이상 고르게 하고 기본은 네 개 다 선택)
features_map = {
    "스크린 수 (상용로그)": "log_first_scrn",
    "누적 관객 (상용로그)": "log_total_audi",
    "10위권 일수": "days_in_top10",
    "롱런 지수": "long_run_idx"
}

selected_labels = st.multiselect(
    "묶는 데 사용할 속성을 골라주세요 (2개 이상 선택):",
    options=list(features_map.keys()),
    default=list(features_map.keys())
)

if len(selected_labels) < 2:
    st.warning("⚠️ 속성을 둘 이상 선택해야 K-평균 분석을 진행할 수 있습니다.")
    st.stop()

selected_features = [features_map[label] for label in selected_labels]

# 5. 데이터 표준화 및 K-Means 군집화 (세 묶음, 난수 고정)
X = df[selected_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster_pred'] = kmeans.fit_predict(X_scaled)

# 누적 관객 평균(원래 단위)이 큰 순서대로 ㉮, ㉯, ㉰ 매핑
cluster_audi_mean = df.groupby('cluster_pred')['total_audi'].mean().sort_values(ascending=False)
cluster_mapping = {cluster_pred: label for cluster_pred, label in zip(cluster_audi_mean.index, ['㉮', '㉯', '㉰'])}
df['cluster'] = df['cluster_pred'].map(cluster_mapping)

# 정렬 순서를 ㉮, ㉯, ㉰ 구조로 고정하기 위한 카테고리 정의
df['cluster'] = pd.Categorical(df['cluster'], categories=['㉮', '㉯', '㉰'], ordered=True)

# 6. 2차원 산점도 (가로축/세로축 속성 선택, 마우스 오버 시 제목 표시)
st.subheader("📈 2차원 시각화 분포")
col1, col2 = st.columns(2)
with col1:
    xaxis = st.selectbox("가로축 속성 선택:", options=selected_labels, index=0)
with col2:
    yaxis = st.selectbox("세로축 속성 선택:", options=selected_labels, index=min(1, len(selected_labels)-1))

fig_2d = px.scatter(
    df,
    x=features_map[xaxis],
    y=features_map[yaxis],
    color='cluster',
    color_discrete_map={'㉮': '#EF553B', '㉯': '#636EFA', '㉰': '#00CC96'},
    category_orders={'cluster': ['㉮', '㉯', '㉰']},
    hover_name='movieNm',
    labels={features_map[xaxis]: xaxis, features_map[yaxis]: yaxis, 'cluster': '묶음'},
    title=f"2D 산점도 분포 ({xaxis} vs {yaxis})"
)
st.plotly_chart(fig_2d, use_container_width=True)

# 7. 3차원 산점도 (속성이 3개 미만일 때 안내 문구 처리, 점 작게, 회전 인터랙션, 마우스 오버 제목)
st.write("---")
if len(selected_features) >= 3:
    st.subheader("🪐 3차원 시각화 분포")
    col3, col4, col5 = st.columns(3)
    with col3:
        z_x = st.selectbox("3D X축 속성 선택:", options=selected_labels, index=0, key="3d_x")
    with col4:
        z_y = st.selectbox("3D Y축 속성 선택:", options=selected_labels, index=min(1, len(selected_labels)-1), key="3d_y")
    with col5:
        z_z = st.selectbox("3D Z축 속성 선택:", options=selected_labels, index=min(2, len(selected_labels)-1), key="3d_z")
    
    fig_3d = px.scatter_3d(
        df,
        x=features_map[z_x],
        y=features_map[z_y],
        z=features_map[z_z],
        color='cluster',
        color_discrete_map={'㉮': '#EF553B', '㉯': '#636EFA', '㉰': '#00CC96'},
        category_orders={'cluster': ['㉮', '㉯', '㉰']},
        hover_name='movieNm',
        labels={features_map[z_x]: z_x, features_map[z_y]: z_y, features_map[z_z]: z_z, 'cluster': '묶음'},
        title="3D 공간 산점도 분포 (드래그로 회전 가능)"
    )
    fig_3d.update_traces(marker=dict(size=3))  # 점 작게 세팅
    st.plotly_chart(fig_3d, use_container_width=True)
else:
    st.info("💡 고른 속성이 셋보다 적으면 3차원 산점도를 그릴 수 없습니다. 속성을 3개 이상 선택해 주세요.")

# 8. 묶음별 편수 및 네 속성의 평균(원래 단위) 표 출력
st.write("---")
st.subheader("📋 묶음별 통계 리포트 (원래 단위 평균)")

summary_df = df.groupby('cluster', observed=False).agg(
    편수=('movieNm', 'count'),
    스크린_수_평균=('first_scrn', 'mean'),
    누적_관객_평균=('total_audi', 'mean'),
    십위권_일수_평균=('days_in_top10', 'mean'),
    롱런_지수_평균=('long_run_idx', 'mean')
).reset_index()

summary_df.columns = ['묶음', '편수', '스크린 수 평균', '누적 관객 평균', '10위권 일수 평균', '롱런 지수 평균']
st.dataframe(summary_df.style.format({
    '편수': '{:,.0f}편',
    '스크린 수 평균': '{:,.1f}개',
    '누적 관객 평균': '{:,.1f}명',
    '10위권 일수 평균': '{:,.1f}일',
    '롱런 지수 평균': '{:,.2f}'
}), use_container_width=True)

# 9. 묶음별 누적 관객 상위 5편 영화 제목 나열
st.subheader("🏆 묶음별 누적 관객 많은 순 Top 5 영화")
top_cols = st.columns(3)

for idx, label in enumerate(['㉮', '㉯', '㉰']):
    with top_cols[idx]:
        st.markdown(f"### 📦 묶음 **{label}**")
        cluster_movies = df[df['cluster'] == label].sort_values(by='total_audi', ascending=False).head(5)
        
        if not cluster_movies.empty:
            for rank, row in enumerate(cluster_movies.itertuples(), 1):
                st.write(f"{rank}. **{row.movieNm}** ({row.total_audi:,}명)")
        else:
            st.write("해당 묶음에 조건에 부합하는 영화가 없습니다.")
